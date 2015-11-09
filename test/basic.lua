local d = require 'autograd'
local tensor = torch.FloatTensor

local x = tensor(2000,100):normal()
local y = tensor(2000):uniform(1.5,10.5):floor()
local yOneHot = d.util.oneHot(y,10)
do
   local lin = d.nn.Linear(100,10)
   local lsm = d.nn.LogSoftMax()
   local lossf = d.nn.ClassNLLCriterion()

   local f = function(params, x, y)
      local h = lin(x, params.W, params.b)
      local yhat = lsm(h)
      local loss = lossf(yhat, y)
      return loss
   end

   local params = {
      W = tensor(10, 100):normal(.01),
      b = tensor(10):zero(),
   }

   local df = d(f)
   for k = 1,20 do
      for i = 1,x:size(1) do
         local grads = df(params, x[i], y[i])
      end
   end

--[[
  local f = function(params, x, y)
      local h1 = torch.tanh( params.W1 * x + params.b1 )
      local h2 = params.W2 * h1 + params.b2
      local loss, yhat = d.loss.crossEntropy(h2, y)
      return loss
   end
--]]

end

