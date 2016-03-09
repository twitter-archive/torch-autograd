-- Tester:
local totem = require 'totem'
local autograd = require 'autograd'
local util = require 'autograd.util'
local gradcheck = require 'autograd.gradcheck' {randomizeInput = true}
local gradcheckConstant = require 'autograd.gradcheck' {randomizeInput = false}
local tester = totem.Tester()
local stringx = require 'pl.stringx'

autograd.protected(true)

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
         return torch.sum( torch.cmul(buffer, buffer) ) / (torch.nDimension(input) == 2 and torch.size(input, 1) * torch.size(input, 2) or torch.size(input, 1))
      end

      local inputSize, outputSize = torch.random(10,100), torch.random(100,1000)
      local inputSize = 24
      local outputSize = 848

      local model = nn.Sequential()
      local linear1 = nn.Linear(inputSize, outputSize):reset()
      local linear2 = nn.Linear(outputSize, inputSize):reset()
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

   AutoModuleLoaded = function()
      local inputSize = 24
      local outputSize = 848
      local version = (jit and 'JIT') or (_VERSION:find('5%.1') and '51') or (_VERSION:find('5%.2') and '52') or (_VERSION:find('5%.3') and '53') or assert('version of Lua not supported: ', _VERSION)
      local mseCriterion = torch.load(sys.fpath()..'/data/criterion.th.'..version)
      local model = torch.load(sys.fpath()..'/data/model.th.'..version)
      local autoModel = torch.load(sys.fpath()..'/data/autoModel.th.'..version)
      local autoMseCriterion = torch.load(sys.fpath()..'/data/autoCriterion.th.'..version)
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

   NNWrapperTableInput = function()
      local A = torch.eye(10)
      local B = torch.eye(10):mul(3)
      local mmFn = autograd.nn.MM()

      local fn = function(inputs)
         return torch.sum(mmFn({inputs.A,inputs.B}))
      end

      tester:assert(gradcheck(fn,{A=A,B=B}), "Incorrect gradient")

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

   Reshape = function()
      local function f(params)
         return torch.sum(torch.reshape(params.x,1,9)*3)
      end
      tester:assert(gradcheck(f, {x=torch.randn(3,3)}), "Incorrect gradient")
   end,

   Clamp = function()
      local W = torch.Tensor(5,25):normal()
      local clampFn = function(inputs)
         return torch.sum(torch.clamp(inputs.W,0,math.huge))
      end
      tester:assert(clampFn({W=W})>0, "Basic sanity check failed")
      tester:assert(gradcheck(clampFn,{W=W}), "Incorrect gradient")
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

   SelfView = function()
      local W = torch.Tensor(5,5):normal()
      local x = torch.Tensor(1,25):normal()

      -- Function:
      local viewFn = function(inputs)
         return torch.sum(inputs.x:view(5,5) + inputs.W)
      end
      local viewAsFn = function(inputs)
         return torch.sum(inputs.x:viewAs(inputs.W) + inputs.W)
      end

      -- Check grads:
      tester:assert(gradcheck(viewFn, {W=W,x=x}), "Incorrect gradient")
      tester:assert(gradcheck(viewAsFn, {W=W,x=x}), "Incorrect gradient")

      -- Check floating point
      gd = autograd(viewFn)({W=W,x=x})
      gf = autograd(viewFn)({W=W:float(),x=x:float()})
      tester:assertTensorEq(gd.W, gf.W:double(), 1e-8, "Incorrect floating point gradient")
      tester:assertTensorEq(gd.x, gf.x:double(), 1e-8, "Incorrect floating point gradient")

      gd = autograd(viewAsFn)({W=W,x=x})
      gf = autograd(viewAsFn)({W=W:float(),x=x:float()})
      tester:assertTensorEq(gd.W, gf.W:double(), 1e-8, "Incorrect floating point gradient")
      tester:assertTensorEq(gd.x, gf.x:double(), 1e-8, "Incorrect floating point gradient")


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

      -- Check floating point
      gd = autograd(viewFn)({W=W,x=x})
      gf = autograd(viewFn)({W=W:float(),x=x:float()})
      tester:assertTensorEq(gd.W, gf.W:double(), 1e-8, "Incorrect floating point gradient")
      tester:assertTensorEq(gd.x, gf.x:double(), 1e-8, "Incorrect floating point gradient")

   end,

   SelfExpand = function()
      local W = torch.Tensor(32,100):normal()
      local x1 = torch.Tensor(1,100):normal()
      local x2 = torch.Tensor(32,1):normal()
      local x3 = torch.Tensor(1,1):normal()

      -- Function:
      local expandFn = function(inputs)
         return torch.sum(torch.sum(inputs.x:expand(32,100) + inputs.W, 2))
      end
      local expandAsFn = function(inputs)
         return torch.sum(torch.sum(inputs.x:expandAs(inputs.W) + inputs.W, 2))
      end

      -- Check grads:
      for ix,x in pairs({x1,x2,x3}) do
         tester:assert(gradcheck(expandFn, {W=W, x=x}), "Incorrect gradient")
         tester:assert(gradcheck(expandAsFn, {W=W, x=x}), "Incorrect gradient")
      end
      -- Check floating point
      for ix,x in pairs({x1,x2,x3}) do
         gd = autograd(expandFn)({W=W,x=x})
         gf = autograd(expandFn)({W=W:float(),x=x:float()})
         tester:assertTensorEq(gd.W, gf.W:double(), 1e-8, "Incorrect floating point gradient")
         tester:assertTensorEq(gd.x, gf.x:double(), 1e-8, "Incorrect floating point gradient")
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
         tester:assert(gradcheck(expandFn, {W=W, x=x}), "Incorrect gradient")
         tester:assert(gradcheck(expandAsFn, {W=W, x=x}), "Incorrect gradient")
      end
      -- Check floating point
      for ix,x in pairs({x1,x2,x3}) do
         gd = autograd(expandFn)({W=W,x=x})
         gf = autograd(expandFn)({W=W:float(),x=x:float()})
         tester:assertTensorEq(gd.W, gf.W:double(), 1e-8, "Incorrect floating point gradient")
         tester:assertTensorEq(gd.x, gf.x:double(), 1e-8, "Incorrect floating point gradient")
      end

 end,
   Transpose = function()
      local fn = function(inputs)
         return torch.sum(torch.t(inputs.x))
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

   GradCheck_Ger = function()
      local A = torch.Tensor(10):normal()
      local B = torch.Tensor(10):normal()
      local func = function(inputs)
         return torch.sum(torch.ger(inputs.A, inputs.B))
      end

      tester:assert(gradcheck(func, {A=A,B=B}), "incorrect gradients")
   end,

   GradCheck_Dot = function()
      -- Parameters:
      local matrices = {
         {torch.Tensor(10,20):normal(), torch.Tensor(20):normal()}, -- 2D x 1D
         {torch.Tensor(10,20):normal(), torch.Tensor(20,1):normal()}, -- 2D x 1D, with second dim
         {torch.Tensor(10,20):normal(), torch.Tensor(20,20):normal()}, -- 2D x 2D
         {torch.Tensor(20,1):normal(), torch.Tensor(1,20):normal()}, -- 1D x 1D
      }

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.A * inputs.B)
      end

      -- Check grads:
      for i,M in pairs(matrices) do
         local A = M[1]
         local B = M[2]
         tester:assert(gradcheck(func, {A=A,B=B}), 'incorrect gradients')
      end
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
      tester:assert(gradcheckConstant(func, {K=K}), 'incorrect gradients')
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
            local minVal = torch[fn](inputs.W, 1)
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

   GradCheck_MLP = function()
      local inputSize = 1024
      local classes = {0,1,2,3,4,5,6,7,8,9}
      -- What model to train:
      local predict,f,params

      -- Define our neural net
      function predict(params, input, target)
         local h1 = torch.tanh(input * params.W[1] + params.B[1])
         local h2 = torch.tanh(h1 * params.W[2] + params.B[2])
         local h3 = h2 * params.W[3] + params.B[3]
         local out = autograd.util.logSoftMax(h3)
         return out
      end

      -- Define our training loss
      function f(params, input, target)
         local prediction = predict(params, input, target)
         local loss = autograd.loss.logMultinomialLoss(prediction, target)
         return loss, prediction
      end

      -- Define our parameters
      -- [-1/sqrt(#output), 1/sqrt(#output)]
      torch.manualSeed(0)
      local W1 = torch.Tensor(inputSize,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
      local B1 = torch.Tensor(50):fill(0)
      local W2 = torch.Tensor(50,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
      local B2 = torch.Tensor(50):fill(0)
      local W3 = torch.Tensor(50,#classes):uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes))
      local B3 = torch.Tensor(#classes):fill(0)

      -- Trainable parameters:
      params = {
         W = {W1, W2, W3},
         B = {B1, B2, B3},
      }

      input = torch.randn(1,1024)
      target = torch.zeros(1,10)
      target[1][3] = 1

      p = {W=params.W,B=params.B, input=input, target=target}

      function testme(params)
         return f({W=params.W,B=params.B}, params.input, params.target)
      end

      tester:assert(gradcheck(testme, p), "Incorrect Gradient")

   end,

   NNFunc_Basic = function()
      -- Tested params:
      local inputSize = 100
      local outputSize = 50
      local x = torch.Tensor(inputSize):normal()

      -- nn modules:
      local linear1, linearParams = autograd.nn.Linear(inputSize, outputSize)
      params = {linearParams=linearParams, x=x}

      -- nn version:
      local function f_nn(params)
         local funcout = linear1(params.linearParams, params.x)
         return torch.sum(funcout)
      end

      -- autograd version:
      local function f_autograd(params)
         return torch.sum(params.linearParams[1] * params.x + params.linearParams[2])
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
      tester:asserteq((grad_nn.linearParams[1]-grad_autograd.linearParams[1]):abs():max(), 0, "Incorrect gradients")
      tester:asserteq((grad_nn.x-grad_autograd.x):abs():max(), 0, "Incorrect gradients")

      -- Run a 2nd time - gradients should get recomputed:
      params.linearParams[1]:normal()
      params.linearParams[2]:normal()
      params.x:normal()

      -- Get the NN predictions
      local pred_nn = f_nn(params)
      local g_nn = autograd(f_nn)
      local grad_nn = g_nn(params)

      -- Get the autograd predictions
      local pred_autograd = f_autograd(params)
      local g_autograd = autograd(f_autograd)
      local grad_autograd = g_autograd(params)

      -- Check
      tester:asserteq((grad_nn.linearParams[1]-grad_autograd.linearParams[1]):abs():max(), 0, "Incorrect gradients")
      tester:asserteq((grad_nn.x-grad_autograd.x):abs():max(), 0, "Incorrect gradients")
   end,

   NNFunc_MLP = function()
      -- More complex model:
      local inputSize = 100
      local hiddenSize = 50
      local outputSize = 10

      -- nn modules and their parameters:
      local params = {}
      local linear1, linear2, acts1, acts2
      linear1, params.linear1 = autograd.nn.Linear(inputSize, hiddenSize)
      acts1 = autograd.nn.Tanh()
      linear2,params.linear2 = autograd.nn.Linear(hiddenSize, outputSize)
      acts2 = autograd.nn.Tanh()

      -- input data
      params.x = torch.Tensor(inputSize):normal()

      -- nn version:
      local function mlp(params)
         local h1 = acts1(linear1(params.linear1, params.x))
         local h2 = acts2(linear2(params.linear2, h1))
         local o = torch.sum(h2)
         return o
      end

      -- Eval:
      local pred = mlp(params)
      local grads = autograd(mlp)(params)

      -- Check grads:
      tester:assert(gradcheck(mlp, params), 'incorrect gradients')
   end,

   NNFunc_CNN = function()
      -- Start params with input data:
      local params = {x=torch.Tensor(3, 8, 8):normal()}

      -- nn modules:
      local conv1, acts1, flatten, linear2, acts2
      conv1,params.conv1 = autograd.nn.SpatialConvolutionMM(3, 16, 3, 3, 1, 1, 1, 1)
      acts1 = autograd.nn.Tanh()
      flatten = autograd.nn.Reshape(16*8*8)
      linear2, params.linear2 = autograd.nn.Linear(16*8*8, 10)
      acts2 = autograd.nn.Tanh()

      -- nn version:
      local function cnn(params)
         local h1 = acts1(conv1(params.conv1, params.x))
         local h2 = acts2(linear2(params.linear2, flatten(h1)))
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

      -- Input data
      local x = torch.FloatTensor(inputSize):normal()

      -- nn modules:
      local linear1, pLinear1 = autograd.nn.Linear(inputSize, hiddenSize)
      local acts1 = autograd.nn.Tanh()
      local linear2, pLinear2 = autograd.nn.Linear(hiddenSize, outputSize)
      local acts2 = autograd.nn.Tanh()
      params = autograd.util.cast({
         linear1 = pLinear1,
         linear2 = pLinear2,
         x = x}, "float")

      -- nn version:
      local function mlp(params)
         local h1 = acts1(linear1(params.linear1, params.x))
         local h2 = acts2(linear2(params.linear2,h1))
         local o = torch.sum(h2)
         return o
      end

      -- Eval:
      local grads = autograd(mlp)(params)
      local pred = mlp(params)

      -- Check grads:
      tester:asserteq(torch.typename(grads.linear1[1]), 'torch.FloatTensor', 'incorrect type')
      tester:asserteq(torch.typename(grads.linear1[2]), 'torch.FloatTensor', 'incorrect type')
      tester:asserteq(torch.typename(grads.linear2[1]), 'torch.FloatTensor', 'incorrect type')
      tester:asserteq(torch.typename(grads.linear2[2]), 'torch.FloatTensor', 'incorrect type')
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
      -- local loss = autograd.nn.MSECriterion()
      local loss = autograd.functionalize(nn.MSECriterion())

      -- Input
      local x = torch.Tensor(3, 8, 8):normal()
      local y = torch.Tensor(10):normal()

      -- Force to double:
      params = autograd.util.cast(params, "double")

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
      local model2 = model:clone()
      model2:zeroGradParameters()
      local yhat = model2:forward(x)
      local gx = model2:backward( x, nn.MSECriterion():backward(yhat,y) )
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

      params[1][1]:normal(0,0.01)
      params[2][1]:normal(0,0.01)

      local i = torch.randn(100)
      local t = torch.Tensor({1,0})

      local l,pred = loss(params, i, t)
      local grads = autograd(loss)(params, i, t)

      tester:asserteq(type(l), 'number', 'loss should be a scalar')
      tester:asserteq(grads[1][1]:dim(), 2, 'weights for layer 2 have incorrect dims')
      tester:asserteq(grads[1][2]:dim(), 1, 'biases for layer 2 have incorrect dims')
      tester:asserteq(grads[2][1]:dim(), 2, 'weights for layer 4 have incorrect dims')
      tester:asserteq(grads[2][2]:dim(), 1, 'biases for layer 4 have incorrect dims')

      -- Gradcheck
      tester:assert(gradcheck(loss, params, i, t), 'incorrect gradients')
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
      params[1][1][1]:normal(0,0.01)
      params[1][2][1]:normal(0,0.01)
      params[2][1][1]:normal(0,0.01)
      params[2][2][1]:normal(0,0.01)

      local i = torch.randn(3,8,8)
      local t = torch.randn(2)

      local l,pred = loss(params, i, t)
      local grads = autograd(loss)(params, i, t)

      tester:asserteq(type(l), 'number', 'loss should be a scalar')

      -- Gradcheck:
      tester:assert(gradcheck(loss, params, i, t), 'incorrect gradients')
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
      tester:assert(gradcheck(loss, params, i), 'incorrect gradients')
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
      tester:assert(gradcheck(loss, params, i), 'incorrect gradients')

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

      -- Test on sequence data:
      local o = loss(params, i)
      local g = autograd(loss)(params, i)

      -- Checks
      tester:asserteq(type(g), 'table', 'gradients could not be computed')
      tester:assert(gradcheck(loss, params, i), 'incorrect gradients')
   end,

   DebuggerDivZero = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.div(inputs.W * inputs.x, 0))
      end

      -- Grads:
      local sawHook = 0
      local badline
      local dFunc = autograd(func, {
         debugHook = function(debugger, msg, gen)
            if sawHook == 0 then
               badline = stringx.split(gen.source, "\n")[gen.line]
               --debugger.showDot()
            end
            sawHook = sawHook + 1
         end
      })

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(sawHook, 5, 'debugHook should have tripped')
      tester:asserteq(badline, "    torch_div(rlocals[2], rlocals[1], 0)", 'debugHook should have showed the bad line')
   end,

   ParamLen = function()
      local params = {torch.Tensor(100):fill(1), torch.Tensor(100):fill(1)}
      -- Function:
      local func = function(params)
         return torch.sum(params[1] + params[2] * #params)
      end

      local df = autograd(func)
      local grads = df(params)

      -- Tests:
      tester:assert(gradcheck(func, params, 1), 'incorrect gradients')

   end,

   -- MissingGradient = function()

   --    -- Function:
   --    local func = function(W)
   --       return torch.sum(torch.reshape(W,5,5,1))
   --    end

   --    local test = function()
   --       return autograd(func)(torch.FloatTensor(5, 5))
   --    end

   --    --test()
   --    local _, msg = pcall(test)
   --    tester:assert(string.find(msg, "missing gradient"), "missing gradient not reported")
   -- end,

   Optim = function()
      local f = function(p, x, y)
         local h1 = torch.tanh(x * p.W[1] + p.b[1])
         return torch.sqrt(torch.sum(torch.pow(y - h1, 2)))
      end

      local df = autograd(f)

      local nData = 5000
      local xs = torch.randn(nData, 10)
      local ys = torch.Tensor(nData, 1)
      for i=1, nData do ys[i][1] = math.tanh(xs[i]:sum()) end

      local learningRate = 1e-3
      local params = {
         W = { torch.randn(10, 1) },
         b = { torch.randn(1) }
      }
      local params3 = {
         W = { params.W[1]:clone() },
         b = { params.b[1]:clone() }
      }

      local loss1
      for e=1, 5 do
         loss1 = 0
         for i=1,nData  do
            local grads, l = df(params, xs:narrow(1, i, 1), ys:narrow(1, i, 1))
            loss1 = loss1 + l / nData
            params.W[1]:add(-learningRate, grads.W[1])
            params.b[1]:add(-learningRate, grads.b[1])
         end
      end

      local state = { learningRate = learningRate }
      local loss3
      for e=1, 5 do
         local optimfn, states = autograd.optim.sgd(df, state, params3)
         loss3 = 0
         for i=1,nData do
            local grads, loss = optimfn(xs:narrow(1, i, 1), ys:narrow(1, i, 1))
            loss3 = loss3 + loss / nData
         end
      end

      tester:asserteq(loss1, loss3, 'sgd wrapper should produce same loss')
   end,

   OptimNN = function()
      local nn = require 'nn'
      local optim = require 'optim'

      torch.manualSeed(0)

      -- Set up the localizer network
      ---------------------------------
      local locnet = nn.Sequential()
      locnet:add(nn.SpatialMaxPooling(2,2,2,2))
      locnet:add(nn.SpatialConvolution(1,20,5,5))
      locnet:add(nn.ReLU(true))
      locnet:add(nn.SpatialMaxPooling(2,2,2,2))
      locnet:add(nn.SpatialConvolution(20,20,5,5))
      locnet:add(nn.ReLU(true))
      locnet:add(nn.View(20*2*2))
      locnet:add(nn.Linear(20*2*2,20))
      locnet:add(nn.ReLU(true))
      locnet:add(nn.Linear(20,6))

      -- Functionalize networks
      ---------------------------------
      local agLocnet, locParams = autograd.functionalize(locnet)

      -- Set up parameters
      ---------------------------------
      params = {
         locParams = locParams,
      }

      -- Define our loss function
      ---------------------------------
      local function f(inputs, bhwdImages, labels)
         local warpPrediction = agLocnet(inputs.locParams, bhwdImages)
         return torch.sum(warpPrediction)
      end

      local g = autograd(f, {optimize = true})

      -- FAILS FOR OTHER OPTIMIZERS AS WELL
      local optimfn, states = autograd.optim.sgd(g, {learningRate=1e-2}, params)

      for i=1,3 do
         -- Get images in BHWD format, labels in one-hot format:
         local data = torch.randn(256,1,32,32)
         local target = torch.zeros(256):random(0,9)

         -- Calculate gradients:
         local grads, loss = optimfn(data, target)

      end
      end,

   NNFunc_WrapWithoutParams = function()
      -- Tests that we can wrap NN modules that do not take parameters
      local tanh = autograd.functionalize(nn.Tanh())
      local a = torch.eye(3)
      tester:assertTensorEq(torch.tanh(a), autograd.nn.Tanh()(a), 1e-8)
      tester:assertTensorEq(torch.tanh(a), tanh(a), 1e-8)
      local loss = autograd.functionalize(nn.MSECriterion())

   end,

   FunctionalizeCriterionModule = function()
      -- Tests the use of table-valued inputs in criterions
      local input = {torch.rand(2,10), torch.randn(2,10)}
      local target = {torch.IntTensor{1,8}, torch.randn(2,10)}
      local nll = nn.ClassNLLCriterion()
      local mse = nn.MSECriterion()
      local pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)
      local output1 = pc:forward(input, target)
      local pcf = autograd.functionalize(pc)
      local mt = getmetatable(pc)
      local output2 = pcf(input, target)
      tester:asserteq(output1, output2, 'loss not equal')
      local f = function(x, y)
         return pcf(x, y)
      end
      tester:assert(gradcheck(f, input, target), 'incorrect gradients')
   end,

   ScalarMul = function()
      -- Tests that functions that use scalar multiply do not cause an error
      function f(params)
        return torch.sum(params.W) * 0.4
      end
      local df = autograd(f)
      local params = { W = torch.randn(5,5)}
      -- this line should not raise an error
      local grads, loss = df(params)
      tester:assert(gradcheck(f, {W=params.W}), 'incorrect gradients')
   end,

   StableGradients = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs, zf)
         local dims = torch.size(zf, 1)
         local w1 = inputs.W
         local x1 = inputs.x
         for i = 1, dims do
            w1 = w1 * 1.1
            x1 = x1 * 1.1
         end
         return torch.sum(w1 * x1 * 3.0 + 1.0)
      end

      local df = autograd(func, { stableGradients = true })
      local g = df({W=W, x=x}, torch.Tensor(1))
      for i = 1, 10 do
         local ng = df({W=W, x=x}, torch.Tensor(i))
         tester:assert(g.W == ng.W, 'gradient tensors should match')
         tester:assert(g.x == ng.x, 'gradient tensors should match')
         local ng = df({W=W, x=x}, torch.Tensor(i))
         tester:assert(g.W == ng.W, 'gradient tensors should match')
         tester:assert(g.x == ng.x, 'gradient tensors should match')
      end
   end,

   LessThan = function()
      local f = function(params, x)
         local s = torch.sum(params.a)
         if autograd.util.lt(s, 3) then
            return s*3
         else
            return s
         end
      end

      tester:assert(gradcheck(f,{a = torch.eye(5)}), "Incorrect gradient")
      tester:assert(gradcheck(f,{a = torch.eye(1)}), "Incorrect gradient")
   end,

   CatNumber = function()
      local function f(params)
         local tbl = {}
         tbl[#tbl+1] = params.a
         tbl[#tbl+1] = params.b
         tbl[#tbl+1] = params.c
         local a = autograd.util.cat(tbl)
         return -torch.sum(a)
      end
      local df = autograd(f)
      local params = {a=1,b=2,c=3}
      local grads, loss = df(params)
      -- It just needs to run, gradcheck doesn't support numbers right now
   end,

   FunctionalFill = function()
      local function f(params)
         local o = util.fill(params.a, torch.sum(params.a))
         return torch.sum(o)
      end
      tester:assert(gradcheck(f,{a = torch.randn(5,5)}), "Incorrect gradient")
   end,

   Padding = function()
      local function adjointSelect(params)
         local padded = autograd.util.selectSliceCopy(params.x, torch.zeros(3,3), 1, 1)
         return torch.sum(padded*3)
      end
      tester:assert(gradcheck(adjointSelect, {x=torch.randn(3)}), "Incorrect gradient")
      local function adjointNarrow(params)
         local padded = autograd.util.narrowSliceCopy(params.x, torch.zeros(3,3), 1, 1, 2)
         return torch.sum(padded*3)
      end
      tester:assert(gradcheck(adjointNarrow, {x=torch.randn(3,2)}), "Incorrect gradient")
      local function adjointIndex(params)
         local padded = autograd.util.indexAdd(params.x, torch.zeros(3,3), 1, torch.LongTensor{3,1})
         return torch.sum(padded*3)
      end
      tester:assert(gradcheck(adjointIndex, {x=torch.randn(2,3)}), "Incorrect gradient")
   end,

   RepeatTensor = function()
      local function f2to2(params)
         local y = torch.repeatTensor(params.x, 2, 2)*3
         return torch.sum(y)
      end
      tester:assert(gradcheck(f2to2, {x=torch.randn(3,3)}), "Incorrect gradient")
      x = torch.randn(3,3)
      local o_double = autograd(f2to2)({x=x}).x
      local o_float = autograd(f2to2)({x=x:float()}).x
      tester:assertTensorEq(o_double, o_float:double(), 1e-10, "Incorrect floating point gradient")

      local function f3to3(params)
         local y = torch.repeatTensor(params.x, 2, 2, 2)*3
         return torch.sum(y)
      end
      tester:assert(gradcheck(f3to3, {x=torch.randn(3,3,3)}), "Incorrect gradient")

      local function f2to3(params)
         local y = torch.repeatTensor(params.x, 2, 2, 2)*3
         return torch.sum(y)
      end
      tester:assert(gradcheck(f2to3, {x=torch.randn(3,3)}), "Incorrect gradient")

      local function f3to4(params)
         local y = torch.repeatTensor(params.x, 2, 2, 2, 2)*3
         return torch.sum(y)
      end
      tester:assert(gradcheck(f3to4, {x=torch.randn(3,3,3)}), "Incorrect gradient")
   end,

   ZeroGrad = function()
      --the output of this function does not depend on params, so its grad should be uniformly zero
      local innerFn = function(params, x, y)
         return torch.norm(torch.add(x,y))
      end

      local dneuralNet = autograd(innerFn)

      local numFeatures = 5
      local testParams = torch.randn(numFeatures)
      local x = torch.randn(numFeatures)
      local y = torch.randn(1)[1]

      local analyticalGrad = testParams:clone():zero()
      local numericalGrad = dneuralNet(testParams,x,y)
      tester:assertTensorEq(analyticalGrad,numericalGrad,1e-8,'analytical and numerical solution do not match')
   end,

   SimpleGradGrad = function()

      local innerFn = function(params, x, y)
         local yHat = params*x
         local squaredLoss = (y - yHat)
         return squaredLoss
      end

      --autodiff
      local dneuralNet = autograd(innerFn)

      --the outer function computes the sum of the gradient of the neural network. Therefore, differentiating yields the sum of each column of the Hessian
      local outerFn = function(params_2,x,y)
         local grad = dneuralNet(params_2,x,y)
         return torch.sum(grad)
      end

      --autodiff solution for sum of each column of Hessian
      local ddf = autograd(outerFn)


      local numFeatures = 1

      local testParams = torch.randn(numFeatures)
      local x = torch.randn(numFeatures)
      local y = torch.randn(1)[1]

      local analyticalGrad = x:clone():mul(-1)
      local numericalGrad = dneuralNet(testParams,x,y)

      tester:assertTensorEq(analyticalGrad,numericalGrad,1e-8,'analytical and numerical solution do not match')

      local analyticalGradGrad = x:clone():zero()
      local numericalGradGrad = ddf(testParams,x,y)

      tester:assertTensorEq(analyticalGradGrad,numericalGradGrad,1e-8,'analytical and numerical solution do not match')
   end,

   GradGrad = function()

      local numFeatures = 5
      local params = torch.randn(numFeatures)

      --synthetic data
      local x = torch.randn(numFeatures)
      local y = torch.randn(1)[1]

      local innerFn = function(params, x, y)
         local yHat = params*x
         local squaredLoss = torch.pow(y - yHat,2)
         return squaredLoss
      end

      --autodiff
      local dneuralNet = autograd(innerFn)
      local numericalGrad = dneuralNet(params,x,y)

      --analytical expression
      local residual = y - params*x
      analyticalGrad = x:clone():mul(-2*residual)

      tester:assertTensorEq(analyticalGrad,numericalGrad,1e-8,'analytical and numerical solution do not match')

      --the outer function computes the sum of the gradient of the neural network. Therefore, differentiating yields the sum of each column of the Hessian
      local outerFn = function(params,x,y)
         local grad = dneuralNet(params,x,y)
         return torch.sum(grad)
      end

      --autodiff solution for sum of each column of Hessian
      local ddf = autograd(outerFn)
      local numericalGradGrad = ddf(params,x,y)

      --analytical expression
      hessian = torch.ger(x,x):mul(2)
      analyticalGradGrad = torch.sum(hessian,2):squeeze()
      tester:assertTensorEq(analyticalGradGrad,numericalGradGrad,1e-8,'analytical and numerical solution do not match')
   end,

   Assignment = function()
      local f1 = function(params)
         local xc = torch.clone(params.x)
         xc[1] = torch.sum(params.y)*2.0
         return torch.sum(xc)
      end
      tester:assert(gradcheck(f1,{x=torch.randn(10),y=torch.randn(3)}), "Incorrect gradient")
      local f2 = function(params)
         local xc = torch.clone(params.x)
         xc[1] = torch.sum(params.y)*2.0
         xc[2] = torch.sum(params.y)*3.0
         return torch.sum(xc)
      end
      tester:assert(gradcheck(f2,{x=torch.randn(10),y=torch.randn(3)}), "Incorrect gradient")
      local f3 = function(params)
         local xc = torch.clone(params.x)
         xc[{1,1}] = torch.sum(params.y)*2.0
         return torch.sum(xc)
      end
      tester:assert(gradcheck(f3,{x=torch.randn(10,10),y=torch.randn(3)}), "Incorrect gradient")
      local f4 = function(params)
         local xc = torch.clone(params.x)
         xc[torch.LongStorage{2,2}] = torch.sum(params.y)
         return torch.sum(xc)
      end
      tester:assert(gradcheck(f4,{x=torch.randn(10,10),y=torch.randn(3)}), "Incorrect gradient")
   end,


}

local function prefixTests(pf, t, skip)
   local nt = { }
   for k, v in pairs(t) do
      if not skip[k] then
         nt[pf .. k] = v
      end
   end
   return nt
end

-- Run tests:
autograd.optimize(true)
tester:add(prefixTests("Optimized_", tests, { })):run()
autograd.optimize(false)
tester = totem.Tester()
tester:add(prefixTests("Direct_", tests, { GradGrad = true, AutoModule = true, DebuggerDivZero = true, StableGradients = true, ZeroGrad = true, SimpleGradGrad = true })):run()

