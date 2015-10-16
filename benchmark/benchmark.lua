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
}

for name,test in pairs(tests) do
   local tnn,tag = test()
   print(c.blue('['..name..']')
      .. ' nn: ' .. c.yellow(tnn) .. ', autograd: ' .. c.red(tag) .. ', ratio: ' .. c.green(tag/tnn..'x'))
end
