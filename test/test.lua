-- Tester:
local _ = require 'moses'
local totem = require 'totem'
local autograd = require 'autograd'
local node = require 'autograd.node'
local Node = node.Node
local isNode = node.isNode
local getValue = node.getValue
local gradcheck = require 'autograd.gradcheck'
local tester = totem.Tester()

-- List of tests:
local tests = {
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
      for iparam,param in pairs({"x", "W"}) do
         tester:assert(gradcheck(selectFn, {W=W,x=x}, param), "Incorrect gradient")
         tester:assert(gradcheck(selectFn2, {W=W,x=x}, param), "Incorrect gradient")
      end
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
      for iparam,param in pairs({"x", "W"}) do
         tester:assert(gradcheck(NarrowFn1D, {W=W,x=x1}, param), "Incorrect gradient")
         tester:assert(gradcheck(NarrowFn2D, {W=W,x=x2}, param), "Incorrect gradient")
      end
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
      for iparam,param in pairs({"x", "W"}) do
         tester:assert(gradcheck(viewFn, {W=W,x=x}, param), "Incorrect gradient")
         tester:assert(gradcheck(viewAsFn, {W=W,x=x}, param), "Incorrect gradient")
      end
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
         for iparam,param in pairs({"x", "W"}) do
            tester:assert(gradcheck(expandFn, {W=W, x=x}, param), "Incorrect gradient")
            tester:assert(gradcheck(expandAsFn, {W=W, x=x}, param), "Incorrect gradient")
         end
      end
   end,

   Cat = function()
      local x1 = torch.Tensor(3,5):normal()
      local x2 = torch.Tensor(7,5):normal()

      -- Function:
      local fn = function(inputs)
         return torch.sum(torch.cat(inputs.x1, inputs.x2, 1))
      end

      -- Check grads:
      for iparam,param in pairs({"x1", "x2"}) do
         tester:assert(gradcheck(fn, {x1=x1, x2=x2}, param), "Incorrect gradient")
      end

      -- Transpose, and cat along the last dim
      local x1 = x1:t():contiguous()
      local x2 = x2:t():contiguous()

      -- Function:
      local fn = function(inputs)
         return torch.sum(torch.cat(inputs.x1, inputs.x2))
      end

      -- Check grads:
      for iparam,param in pairs({"x1", "x2"}) do
         tester:assert(gradcheck(fn, {x1=x1, x2=x2}, param), "Incorrect gradient")
      end
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
      tester:assert(gradcheck(func, {W=W, x=x}, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(func, {W=W, x=x}, 'W'), 'incorrect gradients on W')
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
      pred = f(beta)
      g = autograd(f)
      grad = g(beta)
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(grad:dim(), 2, 'incorrect dims for grad')
   end,
   MinMax = function()
      local fns = {torch.min,torch.max}
      local preds = {{1,5},{2,10}}

      for i=1,2 do
         local W = torch.ones(5,5):fill(2)
         W[1] = 1
         local fn = fns[i]

         local func1 = function(inputs)
            return fn(inputs.W)
         end
         local func2 = function(inputs)
            local minVal,indices = fn(inputs.W, 1)
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
         tester:assert(gradcheck(func1, {W=torch.ones(5,5):fill(2)}, 'W'), 'incorrect gradients on W')

         -- Compute func and grads:
         local grads, pred = dFunc2({W=W})

         -- Tests:
         tester:asserteq(type(pred), 'number', 'incorrect prediction')
         tester:asserteq(pred, preds[i][2], 'incorrect prediction')
         tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
         tester:asserteq(grads.W:size(1), 5, 'incorrect dims for gradients')
         tester:asserteq(grads.W:size(2), 5, 'incorrect dims for gradients')
         tester:assert(gradcheck(func1, {W=torch.ones(5,5):fill(2)}, 'W'), 'incorrect gradients on W')
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
      tester:assert(gradcheck(func, {W=W, x=x}, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(func, {W=W, x=x}, 'W'), 'incorrect gradients on W')
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
      tester:assert(gradcheck(func, {W=W, x=x}, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(func, {W=W, x=x}, 'W'), 'incorrect gradients on W')
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

   NodeClass = function()
      -- Build nodes
      local n = Node:new(3, nil, nil, {})
      local m = Node:new(torch.ones(10), nil, nil, {})

      -- Test we can identify nodes
      tester:asserteq(isNode(n), true, "Did not detect class properly")
      tester:asserteq(isNode(m), true, "Did not detect class properly")
      tester:asserteq(isNode({42}), false, "Did not detect class properly")
      tester:asserteq(isNode(3), false, "Did not detect class properly")
      tester:asserteq(isNode("hey"), false, "Did not detect class properly")
      tester:asserteq(isNode(torch.randn(10)), false, "Did not detect class properly")

      -- TODO: more thorough testing of tables that contain nodes
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
      tester:assert(gradcheck(mlp, params, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(mlp, params, 'W1'), 'incorrect gradients on W1')
      tester:assert(gradcheck(mlp, params, 'b1'), 'incorrect gradients on b1')
      tester:assert(gradcheck(mlp, params, 'W2'), 'incorrect gradients on W2')
      tester:assert(gradcheck(mlp, params, 'b2'), 'incorrect gradients on b2')
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
      tester:assert(gradcheck(cnn, params, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(cnn, params, 'W1'), 'incorrect gradients on W1')
      tester:assert(gradcheck(cnn, params, 'b1'), 'incorrect gradients on b1')
      tester:assert(gradcheck(cnn, params, 'W2'), 'incorrect gradients on W2')
      tester:assert(gradcheck(cnn, params, 'b2'), 'incorrect gradients on b2')
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

      -- Gradcheck doesn't support nested params,
      -- need to do a bit of magic to test it.
      local inputs = {
         W1 = params[1].W,
         b1 = params[1].b,
         W2 = params[2].W,
         b2 = params[2].b,
         x = i,
         y = t,
      }
      local closure = function(inputs)
         local params = {
            {W=inputs.W1, b=inputs.b1},
            {W=inputs.W2, b=inputs.b2},
         }
         return loss(params, inputs.x, inputs.y)
      end
      closure(inputs)
      tester:assert(gradcheck(closure, inputs, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(closure, inputs, 'W1'), 'incorrect gradients on W1')
      tester:assert(gradcheck(closure, inputs, 'b1'), 'incorrect gradients on b1')
      tester:assert(gradcheck(closure, inputs, 'W2'), 'incorrect gradients on W2')
      tester:assert(gradcheck(closure, inputs, 'b2'), 'incorrect gradients on b2')
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

      -- Gradcheck doesn't support nested params,
      -- need to do a bit of magic to test it.
      local inputs = {
         W11 = params1[1].W,
         b11 = params1[1].b,
         W12 = params1[2].W,
         b12 = params1[2].b,
         W21 = params2[1].W,
         b21 = params2[1].b,
         W22 = params2[2].W,
         b22 = params2[2].b,
         x = i,
         y = t,
      }
      local closure = function(inputs)
         local params = {
            {
               {W=inputs.W11, b=inputs.b11},
               {W=inputs.W12, b=inputs.b12},
            },
            {
               {W=inputs.W21, b=inputs.b21},
               {W=inputs.W22, b=inputs.b22},
            },
         }
         return loss(params, inputs.x, inputs.y)
      end
      closure(inputs)
      tester:assert(gradcheck(closure, inputs, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(closure, inputs, 'W11'), 'incorrect gradients on W11')
      tester:assert(gradcheck(closure, inputs, 'b11'), 'incorrect gradients on b11')
      tester:assert(gradcheck(closure, inputs, 'W12'), 'incorrect gradients on W12')
      tester:assert(gradcheck(closure, inputs, 'b12'), 'incorrect gradients on b12')
      tester:assert(gradcheck(closure, inputs, 'W21'), 'incorrect gradients on W21')
      tester:assert(gradcheck(closure, inputs, 'b21'), 'incorrect gradients on b21')
      tester:assert(gradcheck(closure, inputs, 'W22'), 'incorrect gradients on W22')
      tester:assert(gradcheck(closure, inputs, 'b22'), 'incorrect gradients on b22')
   end,

   Models_RecurrentNetwork = function()
      -- Define RNN:
      local f,params = autograd.model.RecurrentNetwork({
         inputFeatures = 10,
         hiddenFeatures = 10,
         outputType = 'last',
      })

      -- Params:
      params[1].Wx:normal(0,0.01)
      params[1].bx:normal(0,0.01)
      params[1].Wh:normal(0,0.01)
      params[1].bh:normal(0,0.01)

      -- Loss
      local loss = function(params, input)
         return torch.sum(f(params, input))
      end

      -- Test on sequence data:
      local i = torch.randn(13, 10)
      local o = loss(params, i)
      local g = autograd(loss)(params, i)

      -- Checks
      tester:asserteq(type(g), 'table', 'gradients could not be computed')

      -- Gradcheck doesn't support nested params,
      -- need to do a bit of magic to test it.
      local inputs = {
         Wx = params[1].Wx,
         bx = params[1].bx,
         Wh = params[1].Wh,
         bh = params[1].bh,
         x = i,
      }
      local closure = function(inputs)
         local params = {
            Wx = inputs.Wx,
            bx = inputs.bx,
            Wh = inputs.Wh,
            bh = inputs.bh,
         }
         return loss(params, inputs.x)
      end
      closure(inputs)
      autograd(closure)(inputs)
      tester:assert(gradcheck(closure, inputs, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(closure, inputs, 'Wx'), 'incorrect gradients on Wx')
      tester:assert(gradcheck(closure, inputs, 'bx'), 'incorrect gradients on bx')
      tester:assert(gradcheck(closure, inputs, 'Wh'), 'incorrect gradients on Wh')
      tester:assert(gradcheck(closure, inputs, 'bh'), 'incorrect gradients on bh')
   end,

   Models_RecurrentLSTMNetwork = function()
      -- Define RNN:
      local f,params = autograd.model.RecurrentLSTMNetwork({
         inputFeatures = 10,
         hiddenFeatures = 10,
         outputType = 'last',
      })

      -- Params:
      params[1].Wx:normal(0,0.01)
      params[1].bx:normal(0,0.01)
      params[1].Wh:normal(0,0.01)
      params[1].bh:normal(0,0.01)

      -- Loss
      local loss = function(params, input)
         return torch.sum(f(params, input))
      end

      -- Test on sequence data:
      local i = torch.randn(13, 10)
      local o = loss(params, i)
      local g = autograd(loss)(params, i)

      -- Checks
      tester:asserteq(type(g), 'table', 'gradients could not be computed')

      -- Gradcheck doesn't support nested params,
      -- need to do a bit of magic to test it.
      local inputs = {
         Wx = params[1].Wx,
         bx = params[1].bx,
         Wh = params[1].Wh,
         bh = params[1].bh,
         x = i,
      }
      local closure = function(inputs)
         local params = {
            Wx = inputs.Wx,
            bx = inputs.bx,
            Wh = inputs.Wh,
            bh = inputs.bh,
         }
         return loss(params, inputs.x)
      end
      closure(inputs)
      autograd(closure)(inputs)
      tester:assert(gradcheck(closure, inputs, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(closure, inputs, 'Wx'), 'incorrect gradients on Wx')
      tester:assert(gradcheck(closure, inputs, 'bx'), 'incorrect gradients on bx')
      tester:assert(gradcheck(closure, inputs, 'Wh'), 'incorrect gradients on Wh')
      tester:assert(gradcheck(closure, inputs, 'bh'), 'incorrect gradients on bh')
   end,
}

-- Run tests:
tester:add(tests):run()
