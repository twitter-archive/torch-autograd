-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'
local getValue = require 'autograd.node'.getValue

-- Load in PENN Treebank dataset
local trainData, valData, testData, dict = require('./get-penn.lua')()
local nTokens = #dict.id2word

-- Max input length to train on
local maxLength = 20

print('loaded data: ', {
   train = trainData,
   validation = valData,
   test = testData,
   nTokens = nTokens,
})

-- Define LSTM layers:
local lstm,params = grad.model.RecurrentLSTMNetwork({
   inputFeatures = 200,
   hiddenFeatures = 200,
   outputType = 'all',
})

-- Complete trainable function:
local f = function(inputs, y, prevState)
   -- Encode all inputs through LSTM:
   local h1,newState = lstm(inputs.params[1], inputs.x, prevState)

   -- Loss:
   local loss = 0
   for i = 1,maxLength do
      -- Classify:
      local h2 = inputs.params[2].W * h1[i] + inputs.params[2].b
      local yhat = grad.util.logSoftMax(h2)
      loss = loss - torch.sum( torch.narrow(yhat, 1, y[i], 1) )
   end

   -- Return avergage loss
   return loss / maxLength, newState
end

-- Cast all to float:
for k,param in pairs(params[1]) do
   params[1][k] = param:float()
end

-- Reset params:
params[1].Wx:normal(0,0.01)
params[1].bx:normal(0,0.01)
params[1].Wh:normal(0,0.01)
params[1].bh:normal(0,0.01)

-- Linear classifier params:
params[2] = {
   W = torch.FloatTensor(#dict.id2word, 200):normal(0,.01),
   b = torch.FloatTensor(#dict.id2word):normal(0,.01),
}

-- Get the gradients closure magically:
local df = grad(f)

-- Word dictionary to train:
local wordVecSize = 200
local words = torch.FloatTensor(nTokens, wordVecSize):normal(0,0.01)

-- Train it
local lr = 1
local aloss = 0
local reportEvery = 100
for epoch = 1,10 do
   print('Training Epoch #'..epoch)
   local lstmState -- clear LSTM state at each new epoch
   for i = 1,trainData:size(1)-maxLength-1,maxLength do
      -- Next sequence:
      local x = trainData:narrow(1,i,maxLength)
      local y = trainData:narrow(1,i+1,maxLength)

      -- Select word vectors
      local xv = words:index(1, x:long())

      -- Grads:
      local vars = {params=params, x=xv}
      local grads,loss,newLstmState = df(vars, y, lstmState)

      -- Preserve state for next iteration
      lstmState = {
         c = getValue(newLstmState.c),
         h = getValue(newLstmState.h),
      }

      -- Update params:
      for i,params in ipairs(params) do
         for k,param in ipairs(params) do
            param:add(-lr, grads[i][k])
         end
      end

      -- Update vectors:
      for i = 1,x:size(1) do
         words[i]:add(-lr, grads.x[i])
      end

      -- Loss: exponentiate nll gives perplexity
      aloss = aloss + loss
      if ((i-1)/maxLength+1) % reportEvery == 0 then
         aloss = aloss / reportEvery
         local perplexity = math.exp(aloss)
         print('average training perplexity = ' .. perplexity)
         aloss = 0
      end
   end
end
