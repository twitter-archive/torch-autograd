-- Options
local opt = lapp [[
Run benchmarks.

Options:
   --type    (default float)    can be: double | float | cuda
]]

-- benchmark of common models
local d = require 'autograd'
local nn = require 'nn'
local c = require 'trepl.colorize'
local getValue = require 'autograd.node'.getValue

-- tic/toc
local tic = sys.tic
local toc = sys.toc
if opt.type == 'cuda' then
   tic = function()
      cutorch.synchronize()
      sys.tic()
   end
   toc = function()
      cutorch.synchronize()
      return sys.toc()
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

-- Test 1: logistic regression
local tests = {
   logistic = function()
      local tnn, tag
      local x = tensor(1000,100):normal()
      local y = tensor(1000):uniform(1.5,10.5):floor()
      local yOneHot = d.util.oneHot(y,10)

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
            local wx = params.W * x + params.b
            local loss, yhat = d.loss.crossEntropy(wx, y)
            return loss
         end
         local params = {
            W = tensor(10, 100):normal(.01),
            b = tensor(10):zero(),
         }

         -- force allocs
         local grads = d(f)(params, x[1], yOneHot[1])

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               local grads = d(f)(params, x[i], yOneHot[i])
            end
         end
         tag = toc()
      end

      return tnn, tag
   end,

   mlp = function()
      local tnn, tag
      local x = tensor(1000,100):normal()
      local y = tensor(1000):uniform(1.5,10.5):floor()
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
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])
         local dloss_dyhat = lossf:backward(yhat, y[1])
         model:backward(x[1], dloss_dyhat)

         tic()
         for i = 1,x:size(1) do
            model:zeroGradParameters()
            local yhat = model:forward(x[i])
            local loss = lossf:forward(yhat, y[i])
            local dloss_dyhat = lossf:backward(yhat, y[i])
            model:backward(x[i], dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local f = function(params, x, y)
            local h1 = torch.tanh( params.W1 * x + params.b1 )
            local h2 = params.W2 * h1 + params.b2
            local loss, yhat = d.loss.crossEntropy(h2, y)
            return loss
         end
         local params = {
            W1 = tensor(1000, 100):normal(.01),
            b1 = tensor(1000):zero(),
            W2 = tensor(10, 1000):normal(.01),
            b2 = tensor(10):zero(),
         }

         -- force allocs
         local grads = d(f)(params, x[1], yOneHot[1])

         tic()
         for i = 1,x:size(1) do
            local grads = d(f)(params, x[i], yOneHot[i])
         end
         tag = toc()
      end

      return tnn, tag
   end,

   mlpHybrid = function()
      local tnn, tag
      local x = tensor(1000,100):normal()
      local y = tensor(1000):uniform(1.5,10.5):floor()
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
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])
         local dloss_dyhat = lossf:backward(yhat, y[1])
         model:backward(x[1], dloss_dyhat)

         tic()
         for i = 1,x:size(1) do
            model:zeroGradParameters()
            local yhat = model:forward(x[i])
            local loss = lossf:forward(yhat, y[i])
            local dloss_dyhat = lossf:backward(yhat, y[i])
            model:backward(x[i], dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local lin1 = d.nn.Linear(100,1000)
         local tanh = d.nn.Tanh()
         local lin2 = d.nn.Linear(1000,10)
         local lsm = d.nn.LogSoftMax()

         local f = function(params, x, y)
            local h1 = tanh( lin1(x, params.W1, params.b1) )
            local h2 = lin2(h1, params.W2, params.b2)
            local yhat = lsm(h2)
            local loss = -torch.sum(torch.cmul(yhat, y))
            return loss
         end
         local params = {
            W1 = tensor(1000, 100):normal(.01),
            b1 = tensor(1000):zero(),
            W2 = tensor(10, 1000):normal(.01),
            b2 = tensor(10):zero(),
         }

         -- force allocs
         local grads = d(f)(params, x[1], yOneHot[1])

         tic()
         for i = 1,x:size(1) do
            local grads = d(f)(params, x[i], yOneHot[i])
         end
         tag = toc()
      end

      return tnn, tag
   end,

   mlpHybridNoOneHot = function()
      local tnn, tag
      local x = tensor(1000,100):normal()
      local y = tensor(1000):uniform(1.5,10.5):floor()

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
         for i = 1,x:size(1) do
            model:zeroGradParameters()
            local yhat = model:forward(x[i])
            local loss = lossf:forward(yhat, y[i])
            local dloss_dyhat = lossf:backward(yhat, y[i])
            model:backward(x[i], dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local lin1 = d.nn.Linear(100,1000)
         local tanh = d.nn.Tanh()
         local lin2 = d.nn.Linear(1000,10)
         local lsm = d.nn.LogSoftMax()

         local f = function(params, x, y)
            local h1 = tanh( lin1(x, params.W1, params.b1) )
            local h2 = lin2(h1, params.W2, params.b2)
            local yhat = lsm(h2)
            local loss = - torch.sum( torch.narrow(yhat, 1, y, 1) )
            return loss
         end
         local params = {
            W1 = tensor(1000, 100):normal(.01),
            b1 = tensor(1000):zero(),
            W2 = tensor(10, 1000):normal(.01),
            b2 = tensor(10):zero(),
         }

         -- force allocs
         local grads = d(f)(params, x[1], y[1])

         tic()
         for i = 1,x:size(1) do
            local grads = d(f)(params, x[i], y[i])
         end
         tag = toc()
      end

      return tnn, tag
   end,

   mlpForward = function()
      local tnn, tag
      local x = tensor(1000,100):normal()
      local y = tensor(1000):uniform(1.5,10.5):floor()
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
         model:zeroGradParameters()
         local yhat = model:forward(x[1])
         local loss = lossf:forward(yhat, y[1])

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               model:zeroGradParameters()
               local yhat = model:forward(x[i])
               local loss = lossf:forward(yhat, y[i])
            end
         end
         tnn = toc()
      end

      do
         local f = function(params, x, y)
            local h1 = torch.tanh( params.W1 * x + params.b1 )
            local h2 = params.W2 * h1 + params.b2
            local loss, yhat = d.loss.crossEntropy(h2, y)
            return loss
         end
         local params = {
            W1 = tensor(1000, 100):normal(.01),
            b1 = tensor(1000):zero(),
            W2 = tensor(10, 1000):normal(.01),
            b2 = tensor(10):zero(),
         }

         -- force allocs
         local grads = f(params, x[1], yOneHot[1])

         tic()
         for k = 1,10 do
            for i = 1,x:size(1) do
               local grads = f(params, x[i], yOneHot[i])
            end
         end
         tag = toc()
      end

      return tnn, tag
   end,

   mlpBatched = function()
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
         for i = 1,100 do
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
            local N = getValue(x):size(1)
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
         local grads = d(f)(params, x, yOneHot)

         tic()
         for i = 1,100 do
            local grads = d(f)(params, x, yOneHot)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   mlpHybridBatched = function()
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
         model:zeroGradParameters()
         local yhat = model:forward(x)
         local loss = lossf:forward(yhat, y)
         local dloss_dyhat = lossf:backward(yhat, y)
         model:backward(x, dloss_dyhat)

         tic()
         for i = 1,100 do
            model:zeroGradParameters()
            local yhat = model:forward(x)
            local loss = lossf:forward(yhat, y)
            local dloss_dyhat = lossf:backward(yhat, y)
            model:backward(x, dloss_dyhat)
         end
         tnn = toc()
      end

      do
         local lin1 = d.nn.Linear(100,1000)
         local tanh = d.nn.Tanh()
         local lin2 = d.nn.Linear(1000,10)
         local lsm = d.nn.LogSoftMax()

         local f = function(params, x, y)
            local h1 = tanh( lin1(x, params.W1, params.b1) )
            local h2 = lin2(h1, params.W2, params.b2)
            local yhat = lsm(h2)
            local loss = -torch.sum(torch.cmul(yhat, y))
            return loss
         end
         local params = {
            W1 = tensor(1000, 100):normal(.01),
            b1 = tensor(1000):zero(),
            W2 = tensor(10, 1000):normal(.01),
            b2 = tensor(10):zero(),
         }

         -- force allocs
         local grads = d(f)(params, x, yOneHot)

         tic()
         for i = 1,100 do
            local grads = d(f)(params, x, yOneHot)
         end
         tag = toc()
      end

      return tnn, tag
   end,

   cnnHybridBatched = function()
      local tnn, tag
      local x = tensor(32,3,64,64):normal()
      local y = tensor(32):uniform(1.5,10.5):floor()
      local yOneHot = d.util.oneHot(y,10)

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
         local c1 = d.nn.SpatialConvolutionMM(3,16,5,5)
         local t1 = d.nn.Tanh()
         local m1 = d.nn.SpatialMaxPooling(2,2,2,2)
         local c2 = d.nn.SpatialConvolutionMM(16,32,5,5)
         local t2 = d.nn.Tanh()
         local m2 = d.nn.SpatialMaxPooling(2,2,2,2)
         local r = d.nn.Reshape(13*13*32)
         local l3 = d.nn.Linear(13*13*32,10)
         local lsm = d.nn.LogSoftMax()

         local f = function(params, x, y)
            local h1 = m1( t1( c1(x, params.W1, params.b1) ) )
            local h2 = m2( t2( c2(h1, params.W2, params.b2) ) )
            local h3 = l3(r(h2), params.W3, params.b3)
            local yhat = lsm(h3)
            local loss = -torch.sum(torch.cmul(yhat, y))
            return loss
         end
         local params = {
            W1 = tensor(16, 3*5*5):normal(.01),
            b1 = tensor(16):zero(),
            W2 = tensor(32, 16*5*5):normal(.01),
            b2 = tensor(32):zero(),
            W3 = tensor(10, 32*13*13):normal(.01),
            b3 = tensor(10):zero(),
         }

         -- force allocs
         local grads = d(f)(params, x, yOneHot)

         tic()
         for i = 1,10 do
            local grads = d(f)(params, x, yOneHot)
         end
         tag = toc()
      end

      return tnn, tag
   end,
}

local fmt = function(nb,color)
   local nb = stringx.rjust(string.format('%.2f',nb), 5)
   return c[color](nb)
end

print('Benchmarks:')
for name,test in pairs(tests) do
   local tnn,tag = test()
   print(c.blue(stringx.rjust('['..name..']', 20))
      .. ' nn: ' .. fmt(tnn,'yellow') .. 's, autograd: ' .. fmt(tag,'red') .. 's, ratio: ' .. fmt(tag/tnn,'green') .. 'x')
end
