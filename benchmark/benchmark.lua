-- benchmark of common models
local d = require 'autograd'
local nn = require 'nn'
local c = require 'trepl.colorize'

-- Test 1: logistic regression
local tests = {
   logistic = function()
      local tnn, tag
      local x = torch.FloatTensor(1000,100):normal()
      local y = torch.FloatTensor(1000):random(1,10)
      local yOneHot = d.util.oneHot(y)

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,10))
         model:add(nn.LogSoftMax())
         model:float()
         local lossf = nn.ClassNLLCriterion()
         lossf:float()

         sys.tic()
         for i = 1,x:size(1) do
            model:zeroGradParameters()
            local yhat = model:forward(x[i])
            local loss = lossf:forward(yhat, y[i])
            local dloss_dyhat = lossf:backward(yhat, y[i])
            model:backward(x[i], dloss_dyhat)
         end
         tnn = sys.toc()
      end

      do
         local f = function(params, x, y)
            local wx = params.W * x + params.b
            local loss, yhat = d.loss.crossEntropy(wx, y)
            return loss
         end
         local params = {
            W = torch.FloatTensor(10, 100):normal(.01),
            b = torch.FloatTensor(10):zero(),
         }

         sys.tic()
         for i = 1,x:size(1) do
            local grads = d(f)(params, x[i], yOneHot[i])
         end
         tag = sys.toc()
      end

      return tnn, tag
   end,

   mlp = function()
      local tnn, tag
      local x = torch.FloatTensor(1000,100):normal()
      local y = torch.FloatTensor(1000):random(1,10)
      local yOneHot = d.util.oneHot(y)

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,1000))
         model:add(nn.Tanh())
         model:add(nn.Linear(1000,10))
         model:add(nn.LogSoftMax())
         model:float()
         local lossf = nn.ClassNLLCriterion()
         lossf:float()

         sys.tic()
         for i = 1,x:size(1) do
            model:zeroGradParameters()
            local yhat = model:forward(x[i])
            local loss = lossf:forward(yhat, y[i])
            local dloss_dyhat = lossf:backward(yhat, y[i])
            model:backward(x[i], dloss_dyhat)
         end
         tnn = sys.toc()
      end

      do
         local f = function(params, x, y)
            local h1 = torch.tanh( params.W1 * x + params.b1 )
            local h2 = params.W2 * h1 + params.b2
            local loss, yhat = d.loss.crossEntropy(h2, y)
            return loss
         end
         local params = {
            W1 = torch.FloatTensor(1000, 100):normal(.01),
            b1 = torch.FloatTensor(1000):zero(),
            W2 = torch.FloatTensor(10, 1000):normal(.01),
            b2 = torch.FloatTensor(10):zero(),
         }

         sys.tic()
         for i = 1,x:size(1) do
            local grads = d(f)(params, x[i], yOneHot[i])
         end
         tag = sys.toc()
      end

      return tnn, tag
   end,

   mlpHybrid = function()
      local tnn, tag
      local x = torch.FloatTensor(1000,100):normal()
      local y = torch.FloatTensor(1000):random(1,10)
      local yOneHot = d.util.oneHot(y)

      do
         local model = nn.Sequential()
         model:add(nn.Linear(100,1000))
         model:add(nn.Tanh())
         model:add(nn.Linear(1000,10))
         model:add(nn.LogSoftMax())
         model:float()
         local lossf = nn.ClassNLLCriterion()
         lossf:float()

         sys.tic()
         for i = 1,x:size(1) do
            model:zeroGradParameters()
            local yhat = model:forward(x[i])
            local loss = lossf:forward(yhat, y[i])
            local dloss_dyhat = lossf:backward(yhat, y[i])
            model:backward(x[i], dloss_dyhat)
         end
         tnn = sys.toc()
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
            W1 = torch.FloatTensor(1000, 100):normal(.01),
            b1 = torch.FloatTensor(1000):zero(),
            W2 = torch.FloatTensor(10, 1000):normal(.01),
            b2 = torch.FloatTensor(10):zero(),
         }

         sys.tic()
         for i = 1,x:size(1) do
            local grads = d(f)(params, x[i], yOneHot[i])
         end
         tag = sys.toc()
      end

      return tnn, tag
   end,
}

for name,test in pairs(tests) do
   local tnn,tag = test()
   print(c.blue('['..name..']')
      .. ' nn: ' .. c.yellow(tnn) .. ', autograd: ' .. c.red(tag) .. ', ratio: ' .. c.green(tag/tnn..'x'))
end
