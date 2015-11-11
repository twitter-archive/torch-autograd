-- Tester:
local totem = require 'totem'
local autograd = require 'autograd'
local gradcheck = require 'autograd.gradcheck'
local tester = totem.Tester()

-- List of tests:
local tests = {
   AutoModule = function()
      local linear  = function(input, weight, bias)
         local y = weight * input + bias
         return y
      end
      local linearReLU  = function(input, weight, bias)
         local y = weight * input + bias
         local output = torch.mul( torch.abs( y ) + y, 0.5)
         return output
      end
      local mse = function(input, target)
         local buffer = input-target
         return torch.sum( torch.cmul(buffer, buffer) ) / (input:dim() == 2 and input:size(1)*input:size(2) or input:size(1))
      end

      local inputSize, outputSize = torch.random(10,100), torch.random(100,1000)


      local model = nn.Sequential()
      local linear1 = nn.Linear(inputSize, outputSize)
      local linear2 = nn.Linear(outputSize, inputSize)
      model:add( linear1 )
      model:add( nn.ReLU() )
      model:add( linear2 )

      local mseCriterion = nn.MSECriterion()
      local autoModel = nn.Sequential()
      local autoLinear1ReLU = autograd.nn.AutoModule('AutoLinearReLU')(linearReLU, linear1.weight:clone(), linear1.bias:clone())
      local autoLinear2 = autograd.nn.AutoModule('AutoLinear')(linear, linear2.weight:clone(), linear2.bias:clone())
      autoModel:add( autoLinear1ReLU )
      autoModel:add( autoLinear2 )
      local autoMseCriterion = autograd.nn.AutoCriterion('AutoMSE')(mse)

      -- Test
      local n = 1000
      local lr = 0.001
      local autoParams,autoGradParams = autoModel:parameters()
      local params,gradParams = model:parameters()
      tester:asserteq(#params == #autoParams and #autoParams == #autoGradParams and #autoGradParams == #gradParams, true, 'Wrong number of parameters/gradients parameters')

      for i=1,n do
         model:zeroGradParameters()
         autoModel:zeroGradParameters()
         local input = torch.Tensor(inputSize):uniform(-5,5)
         local target = input:clone():exp()
         -- Forward
         local output1 = model:forward(input)
         local output2 = autoModel:forward(input)
         local mseOut1 = mseCriterion:forward(output1, target)
         local mseOut2 = autoMseCriterion:forward(output2, target)
         -- Backward
         local gradOutput1 = mseCriterion:backward(output1, target)
         local gradOutput2 = autoMseCriterion:backward(output2, target)
         local gradInput1 = model:backward(input, gradOutput1)
         local gradInput2 = autoModel:backward(input, gradOutput2)
         model:accGradParameters(input, gradOutput1)
         autoModel:accGradParameters(input, gradOutput2)
         for i=1,#autoParams do
            autoParams[i]:add(-lr, autoGradParams[i])
         end
         for i=1,#params do
            params[i]:add(-lr, gradParams[i])
         end
      end
      tester:asserteq((model.modules[1].weight - autoModel.modules[1].weight):abs():max() < 1e-6 , true, "gradient accumulation must be the same.")
      tester:asserteq((model.modules[1].bias - autoModel.modules[1].bias):abs():max() < 1e-6, true, "gradient accumulation must be the same.")
      tester:asserteq((model.modules[3].weight - autoModel.modules[2].weight):abs():max() < 1e-6, true, "gradient accumulation must be the same.")
      tester:asserteq((model.modules[3].bias - autoModel.modules[2].bias):abs():max() < 1e-6, true, "gradient accumulation must be the same.")
   end,
   
   Select = function()
      local W = torch.Tensor(5,25):normal()
      local x = torch.Tensor(1,25):normal()

      -- Function:
      local selectFn = function(inputs)
         return torch.sum(torch.select(inputs.W,1,1) + inputs.x)
      end
      local selectFn2 = function(inputs)
         local a = torch.select(torch.viewAs(torch.select(inputs.W,1,1), inputs.x), 2, 1)
         local b = torch.select(inputs.x, 2, 1)
         return torch.sum(a + b)
      end

      -- Check grads:
      tester:assert(gradcheck(selectFn, {W=W,x=x}), "Incorrect gradient")
      tester:assert(gradcheck(selectFn2, {W=W,x=x}), "Incorrect gradient")
   end,

   Index = function()
      local W = torch.Tensor(5,25):normal()
      local x = torch.Tensor(100,25):normal()

      -- Test with index bigger than the index param + very likely collision of indexes.
      -- i.e. worst case scenario
      local idx = torch.LongTensor(100)
      for i = 1,idx:size(1) do
         idx[i] = torch.random(1,5)
      end

      -- Function:
      local selectFn = function(inputs)
         return torch.sum(torch.index(inputs.W,1,idx) + inputs.x)
      end

      -- Check grads:
      tester:assert(gradcheck(selectFn, {W=W,x=x}), "Incorrect gradient")
   end,

   Narrow = function()
      local W = torch.Tensor(5,25):normal()
      local x1 = torch.Tensor(1,25):normal()
      local x2 = torch.Tensor(3,25):normal()

      -- Function:
      local NarrowFn1D = function(inputs)
         return torch.sum(torch.narrow(inputs.W,1,1,1) + inputs.x)
      end
      local NarrowFn2D = function(inputs)
         return torch.sum(torch.narrow(inputs.W,1,1,3) + inputs.x)
      end

      -- Check grads:
      tester:assert(gradcheck(NarrowFn1D, {W=W,x=x1}), "Incorrect gradient")
      tester:assert(gradcheck(NarrowFn2D, {W=W,x=x2}), "Incorrect gradient")
   end,

   Clone = function()
      local x = torch.Tensor(2,25):normal()

      -- Function:
      local f = function(inputs)
         local res = torch.clone(torch.select(inputs.x, 1, 1) * 10 )
         return torch.sum(res)
      end

      -- Check grads:
      tester:assert(gradcheck(f, {x=x}), "Incorrect gradient")
   end,

   NarrowCopy = function()
      local x = torch.Tensor(2,25):normal()

      -- Function:
      local f = function(inputs)
         local res = inputs.x.new(torch.size(inputs.x))
         local out1 = torch.copy( torch.select(res, 1,1), torch.select(inputs.x, 1, 1) * 10 )
         local out2 = torch.copy( torch.select(res, 1,2), torch.select(inputs.x, 1, 2) * 3 )
         return torch.sum(out1) + torch.sum(out2)
      end

      -- Check grads:
      tester:assert(gradcheck(f, {x=x}), "Incorrect gradient")
   end,

   View = function()
      local W = torch.Tensor(5,5):normal()
      local x = torch.Tensor(1,25):normal()

      -- Function:
      local viewFn = function(inputs)
         return torch.sum(torch.view(inputs.x,5,5) + inputs.W)
      end
      local viewAsFn = function(inputs)
         return torch.sum(torch.viewAs(inputs.x, inputs.W) + inputs.W)
      end

      -- Check grads:
      tester:assert(gradcheck(viewFn, {W=W,x=x}), "Incorrect gradient")
      tester:assert(gradcheck(viewAsFn, {W=W,x=x}), "Incorrect gradient")
   end,

   Expand = function()
      local W = torch.Tensor(32,100):normal()
      local x1 = torch.Tensor(1,100):normal()
      local x2 = torch.Tensor(32,1):normal()
      local x3 = torch.Tensor(1,1):normal()

      -- Function:
      local expandFn = function(inputs)
         return torch.sum(torch.sum(torch.expand(inputs.x, 32, 100) + inputs.W, 2))
      end
      local expandAsFn = function(inputs)
         return torch.sum(torch.sum(torch.expandAs(inputs.x, inputs.W) + inputs.W, 2))
      end

      -- Check grads:
      for ix,x in pairs({x1,x2,x3}) do
         tester:assert(gradcheck(expandFn, {W=W, x=x}), "Incorrect gradient")
         tester:assert(gradcheck(expandAsFn, {W=W, x=x}), "Incorrect gradient")
      end
   end,

   Transpose = function()
      local fn = function(inputs)
         return torch.sum(torch.transpose(inputs.x))
      end

      -- Check grads:
      local x = torch.Tensor(10,5):normal()
      tester:assert(gradcheck(fn, {x=x}), "Incorrect gradient")
   end,

   Cat = function()
      -- Concat along 1st dim:
      local x1 = torch.Tensor(3,5):normal()
      local x2 = torch.Tensor(7,5):normal()

      -- Function:
      local fn = function(inputs)
         return torch.sum(torch.cat(inputs.x1, inputs.x2, 1))
      end

      -- Check grads:
      tester:assert(gradcheck(fn, {x1=x1, x2=x2}), "Incorrect gradient")

      -- Transpose, and cat along the last dim
      local x1 = x1:t():contiguous()
      local x2 = x2:t():contiguous()

      -- Function:
      local fn = function(inputs)
         return torch.sum(torch.cat(inputs.x1, inputs.x2))
      end

      -- Check grads:
      tester:assert(gradcheck(fn, {x1=x1, x2=x2}), "Incorrect gradient")

      -- Tables of tensors
      local xs = {torch.Tensor(10):normal(), torch.Tensor(10):normal(), torch.Tensor(10):normal()}

      -- Function:
      local fn = function(inputs)
         return torch.sum(torch.cat(inputs,1))
      end

      -- Check grads:
      tester:assert(gradcheck(fn, xs), "Incorrect gradient")
   end,

   Dot = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   GradCheck_Dot = function()
      -- Parameters:
      local W = torch.Tensor(32,100):normal()
      local x = torch.Tensor(100):normal()

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Check grads:
      tester:assert(gradcheck(func, {W=W, x=x}), 'incorrect gradients')
   end,

   Inverse = function()
      -- Parameters:
      local x = torch.Tensor(20):fill(.5)
      local K = torch.eye(20) + torch.ger(x,x)

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.inverse(inputs.K))
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({K=K})
      local grads = dFunc({K=K})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, torch.sum(torch.inverse(K)), 'incorrect prediction')
      tester:asserteq(grads.K:dim(), 2, 'incorrect dims for gradients')
   end,

   GradCheck_Inverse = function()
      -- Parameters:
      local x = torch.Tensor(10):normal()
      local K = torch.eye(10) + torch.ger(x,x)

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.inverse(inputs.K))
      end

      -- Check grads:
      tester:assert(gradcheck(func, {K=K}), 'incorrect gradients')
   end,

   Scale = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x * 3.0 + 1.0)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 2432, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   GradientTensorSize = function()

      local f = function(beta)
         -- beta:  2x2
         local maxed = torch.max(beta)
         -- beta: 2x2, maxed: number
         local diff = beta - maxed
         -- diff: 2x2
         local summed = torch.sum(diff, 2)
         -- summed: 2x1, maxed: number
         local out = summed + maxed -- if you comment out maxed, this works
         -- out: 2x1
         return torch.sum(out)
      end

      local beta = torch.eye(2,2)
      local pred = f(beta)
      local g = autograd(f)
      local grad = g(beta)
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(grad:dim(), 2, 'incorrect dims for grad')
   end,

   MinMax = function()
      local fns = {"min", "max"}
      local preds = {{1,5},{2,10}}

      for i=1,2 do
         local W = torch.ones(5,5):fill(2)
         W[1] = 1
         local fn = fns[i]

         local func1 = function(inputs)
            return torch[fn](inputs.W)
         end
         local func2 = function(inputs)
            local minVal,indices = torch[fn](inputs.W, 1)
            return torch.sum(minVal)
         end

         -- Grads:
         local dFunc1 = autograd(func1)
         local dFunc2 = autograd(func2)

         -- Compute func and grads:
         local grads, pred = dFunc1({W=W})

         -- Tests:
         tester:asserteq(type(pred), 'number', 'incorrect prediction')
         tester:asserteq(pred, preds[i][1], 'incorrect prediction')
         tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
         tester:asserteq(grads.W:size(1), 5, 'incorrect dims for gradients')
         tester:asserteq(grads.W:size(2), 5, 'incorrect dims for gradients')
         tester:assert(gradcheck(func1, {W=W}), 'incorrect gradients')

         -- Compute func and grads:
         local W = torch.ones(5,5):fill(2)
         W[1] = 1
         local grads, pred = dFunc2({W=W})

         -- Tests:
         tester:asserteq(type(pred), 'number', 'incorrect prediction')
         tester:asserteq(pred, preds[i][2], 'incorrect prediction')
         tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
         tester:asserteq(grads.W:size(1), 5, 'incorrect dims for gradients')
         tester:asserteq(grads.W:size(2), 5, 'incorrect dims for gradients')
         tester:assert(gradcheck(func1, {W=W}), 'incorrect gradients')
      end
   end,

   GradCheck_Scale = function()
      -- Parameters:
      local W = torch.Tensor(32,100):normal()
      local x = torch.Tensor(100):normal()

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x * 3.0 + 1.0)
      end

      -- Check grads:
      tester:assert(gradcheck(func, {W=W, x=x}), 'incorrect gradients')
   end,

   Unary = function()
      -- Parameters:
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(- inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({x=x})
      local grads = dFunc({x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, -50, 'incorrect prediction')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   DotNonLinear = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.tanh(inputs.W * inputs.x))
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 32, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   GradCheck_DotNonLinear = function()
      -- Parameters:
      local W = torch.Tensor(32,100):normal()
      local x = torch.Tensor(100):normal()

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.tanh(inputs.W * inputs.x))
      end

      -- Check grads:
      tester:assert(gradcheck(func, {W=W, x=x}), 'incorrect gradients')
   end,

   FloatType = function()
      -- Parameters:
      local W = torch.FloatTensor(32,100):fill(.5)
      local x = torch.FloatTensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   CudaType = function()
      -- Cuda only:
      if not cutorch then
         return
      end

      -- Parameters:
      local W = torch.CudaTensor(32,100):fill(.5)
      local x = torch.CudaTensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   NCalls = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})
      for i = 1,5 do
         pred = func({W=W, x=x})
         grads = dFunc({W=W, x=x})
      end

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   NNFunc_Basic = function()
      -- Tested params:
      local inputSize = 100
      local outputSize = 50
      local W = torch.Tensor(outputSize,inputSize):normal()
      local b = torch.Tensor(outputSize):normal()
      local x = torch.Tensor(inputSize):normal()
      local params = {W=W, b=b, x=x}

      -- nn modules:
      local linear1 = autograd.nn.Linear(inputSize, outputSize)

      -- nn version:
      local function f_nn(params)
         local funcout = linear1(params.x, params.W, params.b)
         return torch.sum(funcout)
      end

      -- autograd version:
      local function f_autograd(params)
         return torch.sum(params.W * params.x + params.b)
      end

      -- Get the NN predictions
      local pred_nn = f_nn(params)
      local g_nn = autograd(f_nn)
      local grad_nn = g_nn(params)

      -- Get the autograd predictions
      local pred_autograd = f_autograd(params)
      local g_autograd = autograd(f_autograd)
      local grad_autograd = g_autograd(params)

      -- Check
      tester:asserteq((grad_nn.W-grad_autograd.W):abs():max(), 0, "Incorrect gradients")
      tester:asserteq((grad_nn.x-grad_autograd.x):abs():max(), 0, "Incorrect gradients")

      -- Run a 2nd time - gradients should get recomputed:
      W:normal()
      b:normal()
      x:normal()

      -- Get the NN predictions
      local pred_nn = f_nn(params)
      local g_nn = autograd(f_nn)
      local grad_nn = g_nn(params)

      -- Get the autograd predictions
      local pred_autograd = f_autograd(params)
      local g_autograd = autograd(f_autograd)
      local grad_autograd = g_autograd(params)

      -- Check
      tester:asserteq((grad_nn.W-grad_autograd.W):abs():max(), 0, "Incorrect gradients")
      tester:asserteq((grad_nn.x-grad_autograd.x):abs():max(), 0, "Incorrect gradients")
   end,

   NNFunc_MLP = function()
      -- More complex model:
      local inputSize = 100
      local hiddenSize = 50
      local outputSize = 10

      -- Trainable parameters:
      local x = torch.Tensor(inputSize):normal()
      local W1 = torch.Tensor(hiddenSize,inputSize):normal()
      local b1 = torch.Tensor(hiddenSize):normal()
      local W2 = torch.Tensor(outputSize,hiddenSize):normal()
      local b2 = torch.Tensor(outputSize):normal()
      local params = {W1=W1, b1=b1, W2=W2, b2=b2, x=x}

      -- nn modules:
      local linear1 = autograd.nn.Linear(inputSize, hiddenSize)
      local acts1 = autograd.nn.Tanh()
      local linear2 = autograd.nn.Linear(hiddenSize, outputSize)
      local acts2 = autograd.nn.Tanh()

      -- nn version:
      local function mlp(params)
         local h1 = acts1(linear1(params.x, params.W1, params.b1))
         local h2 = acts2(linear2(h1, params.W2, params.b2))
         local o = torch.sum(h2)
         return o
      end

      -- Eval:
      local pred = mlp(params)
      local grads = autograd(mlp)(params)

      -- Check grads:
      tester:assert(gradcheck(mlp, params, x), 'incorrect gradients')
   end,

   NNFunc_CNN = function()
      -- Trainable parameters:
      local x = torch.Tensor(3, 8, 8):normal()
      local W1 = torch.Tensor(16, 3*3*3):normal()
      local b1 = torch.Tensor(16):normal()
      local W2 = torch.Tensor(10, 16*8*8):normal()
      local b2 = torch.Tensor(10):normal()
      local params = {W1=W1, b1=b1, W2=W2, b2=b2, x=x}

      -- nn modules:
      local conv1 = autograd.nn.SpatialConvolutionMM(3, 16, 3, 3, 1, 1, 1, 1)
      local acts1 = autograd.nn.Tanh()
      local flatten = autograd.nn.Reshape(16*8*8)
      local linear2 = autograd.nn.Linear(16*8*8, 10)
      local acts2 = autograd.nn.Tanh()

      -- nn version:
      local function cnn(params)
         local h1 = acts1(conv1(params.x, params.W1, params.b1))
         local h2 = acts2(linear2(flatten(h1), params.W2, params.b2))
         local o = torch.sum(h2)
         return o
      end

      -- Eval:
      local pred = cnn(params)
      local grads = autograd(cnn)(params)

      -- Check grads:
      tester:assert(gradcheck(cnn, params), 'incorrect gradients')
   end,

   NNFunc_Float = function()
      -- More complex model:
      local inputSize = 100
      local hiddenSize = 50
      local outputSize = 10

      -- Trainable parameters:
      local x = torch.FloatTensor(inputSize):normal()
      local W1 = torch.FloatTensor(hiddenSize,inputSize):normal()
      local b1 = torch.FloatTensor(hiddenSize):normal()
      local W2 = torch.FloatTensor(outputSize,hiddenSize):normal()
      local b2 = torch.FloatTensor(outputSize):normal()
      local params = {W1=W1, b1=b1, W2=W2, b2=b2, x=x}

      -- nn modules:
      local linear1 = autograd.nn.Linear(inputSize, hiddenSize)
      local acts1 = autograd.nn.Tanh()
      local linear2 = autograd.nn.Linear(hiddenSize, outputSize)
      local acts2 = autograd.nn.Tanh()

      -- nn version:
      local function mlp(params)
         local h1 = acts1(linear1(params.x, params.W1, params.b1))
         local h2 = acts2(linear2(h1, params.W2, params.b2))
         local o = torch.sum(h2)
         return o
      end

      -- Eval:
      local grads = autograd(mlp)(params)
      local pred = mlp(params)

      -- Check grads:
      tester:asserteq(torch.typename(grads.W1), 'torch.FloatTensor', 'incorrect type')
      tester:asserteq(torch.typename(grads.W2), 'torch.FloatTensor', 'incorrect type')
      tester:asserteq(torch.typename(grads.b1), 'torch.FloatTensor', 'incorrect type')
      tester:asserteq(torch.typename(grads.b2), 'torch.FloatTensor', 'incorrect type')
   end,

   NNFunc_DynamicWrap = function()
      -- Define regular nn model:
      local model = nn.Sequential()
      model:add(nn.SpatialConvolutionMM(3, 16, 3, 3, 1, 1, 1, 1))
      model:add(nn.Tanh())
      model:add(nn.Reshape(16*8*8))
      model:add(nn.Linear(16*8*8, 10))
      model:add(nn.Tanh())

      -- Functionalize!
      local modelf, params = autograd.functionalize(model)

      -- Loss
      local loss = autograd.nn.MSECriterion()

      -- Input
      local x = torch.FloatTensor(3, 8, 8):normal()
      local y = torch.FloatTensor(10):normal()

      -- Force to float:
      for i,p in ipairs(params) do
         params[i] = p:float()
      end

      -- nn version:
      local function cnn(params, y)
         local h2 = modelf(params, params.x)
         return loss(h2, y)
      end

      -- Eval:
      params.x = x
      local pred = cnn(params, y)
      local grads = autograd(cnn)(params, y)

      -- Clone model to compare to built-in nn grad eval:
      local model2 = model:clone():float()
      model2:zeroGradParameters()
      local yhat = model2:forward(x)
      local gx = model2:backward( x, nn.MSECriterion():float():backward(yhat,y) )
      local _,grads2 = model:parameters()

      -- Check errs:
      for i in ipairs(grads) do
         local err = (grads[i] - grads2[i]):abs():max()
         tester:asserteq(err, 0, 'incorrect grad wrapper')
      end
      local err = (gx - grads.x):abs():max()
      tester:asserteq(err, 0, 'incorrect grad wrapper')
   end,

   Models_NeuralNetwork = function()
      -- Define model:
      local f,params = autograd.model.NeuralNetwork({
         inputFeatures = 100,
         hiddenFeatures = {50,2},
         classifier = true,
      })

      -- Loss:
      local loss = function(params, input, target)
         local pred = f(params, input)
         local loss = autograd.loss.crossEntropy(pred,target)
         return loss,pred
      end

      params[1].W:normal(0,0.01)
      params[2].W:normal(0,0.01)

      local i = torch.randn(100)
      local t = torch.Tensor({1,0})

      local l,pred = loss(params, i, t)
      local grads = autograd(loss)(params, i, t)

      tester:asserteq(type(l), 'number', 'loss should be a scalar')
      tester:asserteq(grads[1].W:dim(), 2, 'weights for layer 2 have incorrect dims')
      tester:asserteq(grads[1].b:dim(), 1, 'biases for layer 2 have incorrect dims')
      tester:asserteq(grads[2].W:dim(), 2, 'weights for layer 4 have incorrect dims')
      tester:asserteq(grads[2].b:dim(), 1, 'biases for layer 4 have incorrect dims')

      -- Gradcheck
      tester:assert(gradcheck(function(params, i, t) return loss(params, i, t) end, params, i, t), 'incorrect gradients')
   end,

   Models_SpatialNetwork = function()
      -- Define conv layers:
      local f1,params1 = autograd.model.SpatialNetwork({
         inputFeatures = 3,
         hiddenFeatures = {16, 16},
         poolings = {4, 2},
         kernelSize = 3,
         activations = 'Tanh',
      })

      -- Define upper regular layers:
      local f2,params2 = autograd.model.NeuralNetwork({
         inputFeatures = 16,
         hiddenFeatures = {32,2},
         classifier = true,
      })

      -- Loss == full conv-net with least-squares loss:
      local loss = function(params, input, target)
         local conv = f1(params[1], input)
         local pred = f2(params[2], conv)
         local loss = autograd.loss.leastSquares(pred,target)
         return loss,pred
      end

      local params = {params1, params2}
      params[1][1].W:normal(0,0.01)
      params[1][2].W:normal(0,0.01)
      params[2][1].W:normal(0,0.01)
      params[2][2].W:normal(0,0.01)

      local i = torch.randn(3,8,8)
      local t = torch.randn(2)

      local l,pred = loss(params, i, t)
      local grads = autograd(loss)(params, i, t)

      tester:asserteq(type(l), 'number', 'loss should be a scalar')

      -- Gradcheck:
      tester:assert(gradcheck(function(params, i, t) return loss(params, i, t) end, params, i, t), 'incorrect gradients')
   end,

   Models_RecurrentNetwork = function()
      -- Define RNN:
      local f,params = autograd.model.RecurrentNetwork({
         inputFeatures = 10,
         hiddenFeatures = 10,
         outputType = 'last',
      })

      -- Params:
      params[1].W:normal(0,0.01)
      params[1].b:normal(0,0.01)

      -- Loss
      local loss = function(params, input)
         local v = f(params, input)
         return torch.sum(v)
      end

      -- Test on sequence data:
      local i = torch.randn(13, 10)
      local o = loss(params, i)
      local g = autograd(loss)(params, i)

      -- Checks
      tester:asserteq(type(g), 'table', 'gradients could not be computed')

      -- Gradcheck:
      tester:assert(gradcheck(function(params, i) return loss(params, i) end, params, i), 'incorrect gradients')
   end,

   Models_RecurrentLSTMNetwork = function()
      -- Define RNN:
      local f,params = autograd.model.RecurrentLSTMNetwork({
         inputFeatures = 10,
         hiddenFeatures = 10,
         outputType = 'last',
      })

      -- Params:
      params[1].W:normal(0,0.01)
      params[1].b:normal(0,0.01)

      -- Loss
      local loss = function(params, input)
         local v = f(params, input)
         return torch.sum(v)
      end

      -- Test on sequence data:
      local i = torch.randn(13, 10)
      local o = loss(params, i)
      local g = autograd(loss)(params, i)

      -- Checks
      tester:asserteq(type(g), 'table', 'gradients could not be computed')

      -- Gradcheck:
      tester:assert(gradcheck(function(params, i) return loss(params, i) end, params, i), 'incorrect gradients')

      -- Define RNN with all states exposed:
      local f,params = autograd.model.RecurrentLSTMNetwork({
         inputFeatures = 10,
         hiddenFeatures = 10,
         outputType = 'all',
      })

      -- Loss
      local loss = function(params, input)
         local v = f(params, input)
         return torch.sum(v)
      end

      -- Move to Float
      params[1].W = params[1].W:float()
      params[1].b = params[1].b:float()
      i = i:float()

      -- Test on sequence data:
      local o = loss(params, i)
      local g = autograd(loss)(params, i)

      -- Checks
      tester:asserteq(type(g), 'table', 'gradients could not be computed')
   end,
}

-- Run tests:
tester:add(tests):run()
