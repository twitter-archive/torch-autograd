require 'trepl'

-- Options
local opt = lapp [[
Run benchmarks.

Options:
   --type    (default float)    can be: double | float | cuda
   --nodes   (default false)
   --profile (default false)    requires profi to be installed (luarocks install profi)
   --nooptimize (default false)
]]

-- benchmark of common models
local d = require 'autograd'
local nn = require 'nn'
local c = require 'trepl.colorize'
local haveProfi,profi = pcall(require,'ProFi')

d.optimize(opt.nooptimize == 'false')

-- tic/toc
local tic,toc
local tm = torch.Timer()
if opt.type == 'cuda' then
   tic = function()
      cutorch.synchronize()
      tm:reset()
   end
   toc = function()
      cutorch.synchronize()
      return tm:time().real
   end
else
   tic = function()
      tm:reset()
   end
   toc = function()
      return tm:time().real
   end
end

-- type
local tensor = torch.FloatTensor
local ttype = 'torch.FloatTensor'
if opt.type == 'cuda' then
   require 'cunn'
   tensor = torch.CudaTensor
   ttype = 'torch.CudaTensor'
elseif opt.type == 'double' then
   tensor = torch.DoubleTensor
   ttype = 'torch.DoubleTensor'
end

local nodeTimes = { }

if opt.nodes ~= 'false' then
   local preTime;
   d.debugFns.preGradFn = function(node)
      preTime = sys.clock()
   end
   d.debugFns.postGradFn = function(node)
      if node.gradFun ~= nil then
         local idx = 'grad ' .. node.gradFun[1]
         nodeTimes[idx] = (nodeTimes[idx] or 0) + (sys.clock() - preTime)
      end
   end
   local preTime;
   d.debugFns.preFwdFn = function(fn)
      return sys.clock()
   end
   d.debugFns.postFwdFn = function(fn, o)
      local idx = 'forward ' .. fn
      local tm = (sys.clock() - o)
      nodeTimes['forward (inclusive)'] = (nodeTimes['forward (inclusive)'] or 0) + tm
      nodeTimes[idx] = (nodeTimes[idx] or 0) + tm
   end
end

-- Test 1: logistic regression
local tests = {

   ['logistic (ag)'] = function()
      local tnn, tag
      local x = tensor(1000,100):normal()
      local y = tensor(1000):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])
         local dloss_dyhat = lossf:backward(yhat, y[1])
         model:backward(x[1], dloss_dyhat)

         tic()
         for k = 1,40 do
            for i = 1,x:size(1) do
               model:zeroGradParameters()
               local yhat = model:forward(x[i])
               local loss = lossf:forward(yhat, y[i])
               local dloss_dyhat = lossf:backward(yhat, y[i])
               model:backward(x[i], dloss_dyhat)
            end
         end
         tnn = toc()
      end

      do
         local f = function(params, x, y)
            local wx = params.W * x + params.b
            local yhat = d.util.logSoftMax(wx)
            local loss = -torch.sum(torch.narrow(yhat,1,y,1))
            return loss
         end
         local params = {
            W = tensor(10, 100):normal(.01),
            b = tensor(10):zero(),
         }

         -- force allocs
         local df = d(f)
         local grads = df(params, x[1], y[1])

         tic()
         for k = 1,40 do
            for i = 1,x:size(1) do
               local grads = df(params, x[i], y[i])
            end
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['logistic (nn)'] = function()
      local tnn, tag
      local x = tensor(1000,100):normal()
      local y = tensor(1000):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])
         local dloss_dyhat = lossf:backward(yhat, y[1])
         model:backward(x[1], dloss_dyhat)

         tic()
         for k = 1,20 do
            for i = 1,x:size(1) do
               model:zeroGradParameters()
               local yhat = model:forward(x[i])
               local loss = lossf:forward(yhat, y[i])
               local dloss_dyhat = lossf:backward(yhat, y[i])
               model:backward(x[i], dloss_dyhat)
            end
         end
         tnn = toc()
      end

      do
         local lin
         local params = {}
         lin,params.lin = d.nn.Linear(100,10)
         local lsm = d.nn.LogSoftMax()
         local lossf = d.nn.ClassNLLCriterion()
         params = d.util.cast(params, ttype)

         local f = function(params, x, y)
            local h = lin(params.lin, x)
            local yhat = lsm(h)
            local loss = lossf(yhat, y)
            return loss
         end

         -- force allocs
         local df = d(f)
         local grads = df(params, x[1], y[1])

         tic()
         for k = 1,20 do
            for i = 1,x:size(1) do
               local grads = df(params, x[i], y[i])
            end
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['mlp (ag)'] = function()
      local tnn, tag
      local x = tensor(2000,100):normal()
      local y = tensor(2000):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,1000))
         model:add(nn.Tanh())
         model:add(nn.Linear(1000,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])
         local dloss_dyhat = lossf:backward(yhat, y[1])
         model:backward(x[1], dloss_dyhat)

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               model:zeroGradParameters()
               local yhat = model:forward(x[i])
               local loss = lossf:forward(yhat, y[i])
               local dloss_dyhat = lossf:backward(yhat, y[i])
               model:backward(x[i], dloss_dyhat)
            end
         end
         tnn = toc()
      end

      do
         local f = function(params, x, y)
            local h1 = torch.tanh( params.W1 * x + params.b1 )
            local h2 = params.W2 * h1 + params.b2
            local yhat = d.util.logSoftMax(h2)
            local loss = -torch.sum(torch.narrow(yhat,1,y,1))
            return loss
         end
         local params = {
            W1 = tensor(1000, 100):normal(.01),
            b1 = tensor(1000):zero(),
            W2 = tensor(10, 1000):normal(.01),
            b2 = tensor(10):zero(),
         }

         -- force allocs
         local df = d(f)
         local grads = df(params, x[1], y[1])

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               local grads = df(params, x[i], y[i])
            end
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['mlp (nn+ag)'] = function()
      local tnn, tag
      local x = tensor(2000,100):normal()
      local y = tensor(2000):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,1000))
         model:add(nn.Tanh())
         model:add(nn.Linear(1000,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])
         local dloss_dyhat = lossf:backward(yhat, y[1])
         model:backward(x[1], dloss_dyhat)

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               model:zeroGradParameters()
               local yhat = model:forward(x[i])
               local loss = lossf:forward(yhat, y[i])
               local dloss_dyhat = lossf:backward(yhat, y[i])
               model:backward(x[i], dloss_dyhat)
            end
         end
         tnn = toc()
      end

      do
         local lin1,tanh,lin2,lsm
         local params = {}
         lin1,params.lin1 = d.nn.Linear(100,1000)
         tanh = d.nn.Tanh()
         lin2,params.lin2 = d.nn.Linear(1000,10)
         lsm = d.nn.LogSoftMax()
         params = d.util.cast(params, ttype)

         local f = function(params, x, y)
            local h1 = tanh( lin1(params.lin1, x) )
            local h2 = lin2(params.lin2, h1)
            local yhat = lsm(h2)
            local loss = -torch.sum(torch.narrow(yhat, 1, y, 1))
            return loss
         end

         -- force allocs
         local df = d(f)
         local grads = df(params, x[1], y[1])

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               local grads = df(params, x[i], y[i])
            end
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['mlp (nn)'] = function()
      local tnn, tag
      local x = tensor(2000,100):normal()
      local y = tensor(2000):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,1000))
         model:add(nn.Tanh())
         model:add(nn.Linear(1000,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])
         local dloss_dyhat = lossf:backward(yhat, y[1])
         model:backward(x[1], dloss_dyhat)

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               model:zeroGradParameters()
               local yhat = model:forward(x[i])
               local loss = lossf:forward(yhat, y[i])
               local dloss_dyhat = lossf:backward(yhat, y[i])
               model:backward(x[i], dloss_dyhat)
            end
         end
         tnn = toc()
      end

      do
         local lin1, tanh, lin2, lsm, lossf
         local params = {}
         lin1,params.lin1 = d.nn.Linear(100,1000)
         tanh = d.nn.Tanh()
         lin2,params.lin2 = d.nn.Linear(1000,10)
         lsm = d.nn.LogSoftMax()
         lossf = d.nn.ClassNLLCriterion()
         params = d.util.cast(params, ttype)

         local f = function(params, x, y)
            local h1 = tanh( lin1(params.lin1, x) )
            local h2 = lin2(params.lin2, h1)
            local yhat = lsm(h2)
            local loss = lossf(yhat, y)
            return loss
         end

         -- force allocs
         local df = d(f)
         local grads = df(params, x[1], y[1])

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               local grads = df(params, x[i], y[i])
            end
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['mlp (autoModule, batched)'] = function()
      local tnn, tag
      local inputSize, outputSize = 100,1000
      local x = tensor(32,inputSize):uniform(-5,5)
      local uniformMultiplier = torch.expand( tensor(inputSize):uniform():resize(1, inputSize), 32, inputSize)
      local y = x:clone():exp():cmul(uniformMultiplier)


      do
         local model = nn.Sequential()
         local linear1 = nn.Linear(inputSize, outputSize)
         local linear2 = nn.Linear(outputSize, inputSize)
         model:add( linear1 )
         model:add( nn.ReLU() )
         model:add( linear2 )
         model:type(ttype)
         local lossf = nn.MSECriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,200 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local linear  = function(input, weight, bias)
            local y = input * weight + torch.expand(bias, torch.size(input, 1), torch.size(bias, 2))
            return y
         end
         local linearReLU  = function(input, weight, bias)
            local y = input * weight + torch.expand(bias, torch.size(input, 1), torch.size(bias, 2))
            local output = torch.mul( torch.abs( y ) + y, 0.5)
            return output
         end
         local mse = function(input, target)
            local buffer = input-target
            return torch.sum( torch.cmul(buffer, buffer) ) / (torch.nDimension(input) == 2 and torch.size(input, 1) * torch.size(input, 2) or torch.size(input, 1))
         end
         local autoModel = nn.Sequential()
         local autoLinear1ReLU = d.nn.AutoModule('AutoLinearReLU')(linearReLU, tensor(inputSize, outputSize), tensor(1,outputSize))
         local autoLinear2 = d.nn.AutoModule('AutoLinear')(linear, tensor(outputSize, inputSize), tensor(1,inputSize))
         autoModel:add( autoLinear1ReLU )
         autoModel:add( autoLinear2 )
         local lossf = d.nn.AutoCriterion('AutoMSE')(mse)

         -- force allocs
         autoModel:zeroGradParameters()
         local yhat = autoModel:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         autoModel:backward(x, dloss_dyhat)

         tic()
         for i = 1,200 do
            autoModel:zeroGradParameters()
            local yhat = autoModel:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            autoModel:backward(x, dloss_dyhat)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['mlp (ag, batched)'] = function()
      local tnn, tag
      local x = tensor(32,100):normal()
      local y = tensor(32):uniform(1.5,10.5):floor()
      local yOneHot = d.util.oneHot(y,10)

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,1000))
         model:add(nn.Tanh())
         model:add(nn.Linear(1000,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,2000 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local f = function(params, x, y)
            local N = torch.size(x, 1)
            local h1 = torch.tanh( x * params.W1 + torch.expand(params.b1, N,1000) )
            local h2 = h1 * params.W2 + torch.expand(params.b2, N,10)
            local loss, yhat = d.loss.crossEntropy(h2, y)
            return loss
         end
         local params = {
            W1 = tensor(1000, 100):normal(.01):t(),
            b1 = tensor(1, 1000):zero(),
            W2 = tensor(10, 1000):normal(.01):t(),
            b2 = tensor(1, 10):zero(),
         }

         -- force allocs
         local df = d(f)
         local grads = df(params, x, yOneHot)

         tic()
         for i = 1,2000 do
            local grads = df(params, x, yOneHot)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['mlp (nn, batched)'] = function()
      local tnn, tag
      local x = tensor(32,1000):normal()
      local y = tensor(32):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(nn.Linear(1000,1000))
         model:add(nn.Tanh())
         model:add(nn.Linear(1000,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,200 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local lin1,tanh,lin2,lsm,lossf
         local params = {}
         lin1,params.lin1 = d.nn.Linear(1000,1000)
         tanh = d.nn.Tanh()
         lin2,params.lin2 = d.nn.Linear(1000,10)
         lsm = d.nn.LogSoftMax()
         lossf = d.nn.ClassNLLCriterion()
         params = d.util.cast(params, ttype)

         local f = function(params, x, y)
            local h1 = tanh( lin1(params.lin1, x) )
            local h2 = lin2(params.lin2, h1)
            local yhat = lsm(h2)
            local loss = lossf(yhat, y)
            return loss
         end

         -- force allocs
         local df = d(f)
         local grads = df(params, x, y)

         tic()
         for i = 1,200 do
            local grads = df(params, x, y)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['cnn (nn, batched)'] = function()
      local tnn, tag
      local x = tensor(32,3,64,64):normal()
      local y = tensor(32):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(nn.SpatialConvolutionMM(3,16,5,5))
         model:add(nn.Tanh())
         model:add(nn.SpatialMaxPooling(2,2,2,2))
         model:add(nn.SpatialConvolutionMM(16,32,5,5))
         model:add(nn.Tanh())
         model:add(nn.SpatialMaxPooling(2,2,2,2))
         model:add(nn.Reshape(13*13*32))
         model:add(nn.Linear(13*13*32,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,10 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local c1,t1,m1,c2,t2,m2,r,l3,lsm,lossf
         local params = {}
         c1,params.c1 = d.nn.SpatialConvolutionMM(3,16,5,5)
         t1 = d.nn.Tanh()
         m1 = d.nn.SpatialMaxPooling(2,2,2,2)
         c2,params.c2 = d.nn.SpatialConvolutionMM(16,32,5,5)
         t2 = d.nn.Tanh()
         m2 = d.nn.SpatialMaxPooling(2,2,2,2)
         r = d.nn.Reshape(13*13*32)
         l3,params.l3 = d.nn.Linear(13*13*32,10)
         lsm = d.nn.LogSoftMax()
         lossf = d.nn.ClassNLLCriterion()
         params = d.util.cast(params, ttype)

         local f = function(params, x, y)
            local h1 = m1( t1( c1(params.c1, x) ) )
            local h2 = m2( t2( c2(params.c2, h1) ) )
            local h3 = l3(params.l3, r(h2))
            local yhat = lsm(h3)
            local loss = lossf(yhat, y)
            return loss
         end
         -- local params = {
         --    W1 = tensor(16, 3*5*5):normal(.01),
         --    b1 = tensor(16):zero(),
         --    W2 = tensor(32, 16*5*5):normal(.01),
         --    b2 = tensor(32):zero(),
         --    W3 = tensor(10, 32*13*13):normal(.01),
         --    b3 = tensor(10):zero(),
         -- }

         -- force allocs
         local df = d(f)
         local grads = df(params, x, y)

         tic()
         for i = 1,10 do
            local grads = df(params, x, y)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['cnn (nnWrap, batched)'] = function()
      local tnn, tag
      local x = tensor(32,3,64,64):normal()
      local y = tensor(32):uniform(1.5,10.5):floor()

      local model = nn.Sequential()
      model:add(nn.SpatialConvolutionMM(3,16,5,5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.SpatialConvolutionMM(16,32,5,5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.Reshape(13*13*32))
      model:add(nn.Linear(13*13*32,10))
      model:add(nn.LogSoftMax())
      model:type(ttype)
      local lossf = nn.ClassNLLCriterion()
      lossf:type(ttype)

      do
         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,10 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local modelf,params = d.functionalize(model)
         local lossf = d.nn.ClassNLLCriterion()

         local f = function(params, x, y)
            local yhat = modelf(params, x)
            local loss = lossf(yhat, y)
            return loss
         end

         -- force allocs
         local df = d(f)
         local grads = df(params, x, y)

         tic()
         for i = 1,10 do
            local grads = df(params, x, y)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['lstm (ag+nn)'] = function()
      -- Depends on CXNN reference implementation
      local ok,cxnn = pcall(require, 'cxnn')
      if not ok then
         return
      end

      -- Data
      local tnn, tag
      local x = tensor(1,33,100):normal()
      local y = tensor(1):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(cxnn.RecurrentLSTMNetwork({
            inputSize = 100,
            hiddenFeatures = {200},
            outputType = 'last',
         }))
         model:add(nn.Linear(200,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,200 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local lstm1
         local params = {}
         lstm1,params.lstm1 = d.model.RecurrentLSTMNetwork({
            inputFeatures = 100,
            hiddenFeatures = 200,
            outputType = 'last',
         })
         local lin2,lsm,lossf
         lin2,params.lin2 = d.nn.Linear(200,10)
         lsm = d.nn.LogSoftMax()
         lossf = d.nn.ClassNLLCriterion()

         local f = function(params, x, y)
            local h1 = lstm1(params.lstm1, x)
            local h2 = lin2(params.lin2, h1)
            local yhat = lsm(h2)
            local loss = lossf(yhat, y)
            return loss
         end

         params = d.util.cast(params, ttype)

         -- force allocs
         local df = d(f)
         local grads = df(params, x, y)

         tic()
         for i = 1,200 do
            local grads = df(params, x, y)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   ['lstm (ag+nn, batched)'] = function()
      -- Depends on CXNN reference implementation
      local ok,cxnn = pcall(require, 'cxnn')
      if not ok then
         return
      end

      -- Data
      local tnn, tag
      local x = tensor(32,33,100):normal()
      local y = tensor(32):uniform(1.5,10.5):floor()

      do
         local model = nn.Sequential()
         model:add(cxnn.RecurrentLSTMNetwork({
            inputSize = 100,
            hiddenFeatures = {200},
            outputType = 'last',
         }))
         model:add(nn.Linear(200,10))
         model:add(nn.LogSoftMax())
         model:type(ttype)
         local lossf = nn.ClassNLLCriterion()
         lossf:type(ttype)

         -- force allocs
         model:zeroGradParameters()
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,30 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local lstm1
         local params = {}
         lstm1,params.lstm1 = d.model.RecurrentLSTMNetwork({
            inputFeatures = 100,
            hiddenFeatures = 200,
            outputType = 'last',
         })
         local lin2, lsm, lossf
         lin2,params.lin2 = d.nn.Linear(200,10)
         lsm = d.nn.LogSoftMax()
         lossf = d.nn.ClassNLLCriterion()
         params = d.util.cast(params, ttype)

         local f = function(params, x, y)
            local h1 = lstm1(params.lstm1, x)
            local h2 = lin2(params.lin2, h1)
            local yhat = lsm(h2)
            local loss = lossf(yhat, y)
            return loss
         end

         for i in ipairs(params) do
            for k in pairs(params[i]) do
               params[i][k] = params[i][k]:type(ttype):normal()
            end
         end

         -- force allocs
         local df = d(f)
         local grads = df(params, x, y)

         tic()
         for i = 1,30 do
            local grads = df(params, x, y)
         end
         tag = toc()
      end

      return tnn, tag
   end
}

local fmt = function(nb,color)
   local nb = stringx.rjust(string.format('%.2f',nb), 5)
   return c[color](nb)
end

function keysSortedByValue(tbl, sortFunction)
  local keys = {}
  for key in pairs(tbl) do
    table.insert(keys, key)
  end
  table.sort(keys, function(a, b)
    return tbl[a] > tbl[b]
  end)
  return keys
end

print('Benchmarks:')
for name,test in pairs(tests) do
   nodeTimes = { }
   if opt.profile ~= 'false' and haveProfi then
      profi:start()
   end
   local tnn,tag = test()
   tnn = tnn or 1/0
   tag = tag or 1/0
   if tnn ~= 1/0 then
      if opt.profile ~= 'false' and haveProfi then
         profi:stop()
         profi:writeReport(string.format("%s.profile.txt",name))
         profi:reset()
      end
      print(c.blue(stringx.rjust(''..name..' => ', 32))
         .. ' nn: ' .. fmt(tnn,'yellow') .. 's, autograd: ' .. fmt(tag,'red') .. 's, ratio: ' .. fmt(tag/tnn,'green') .. 'x')
      if opt.nodes ~= 'false' then
         local sortedKeys = keysSortedByValue(nodeTimes)
         for i, v in pairs(sortedKeys) do
            print(stringx.rjust(v, 41) .. ': ' .. fmt(nodeTimes[v],'red') .. 's')
         end
         print('')
      end
   end
end

