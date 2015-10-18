-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'

-- Load in PENN Treebank dataset
local trainData, valData, testData, dict = require('./get-penn.lua')()
local nTokens = #dict.id2word

print('loaded data: ', {
   train = trainData,
   validation = valData,
   test = testData,
   nTokens = nTokens,
})

-- Define LSTM layers:
local lstm1,params1 = grad.model.RecurrentLSTMNetwork({
   inputFeatures = 100,
   hiddenFeatures = 10,
   outputType = 'last',
})

-- Define linear clsasifier
local linear2,params2 = grad.model.NeuralNetwork({
   inputFeatures = 100,
   hiddenFeatures = { #dict.id2word },
   classifier = true,
})

-- Complete trainable function:
local f = function(inputs, y)
   local h1 = lstm1(inputs.params1, inputs.x)
   local pred = linear2(inputs.params2, h1)
   local yhat = grad.util.logSoftMax(pred)
   local loss = - torch.sum( torch.narrow(yhat, 1, y, 1) )
   return loss,yhat
end

-- Reset params:
params1[1].Wx:normal(0,0.01)
params1[1].bx:normal(0,0.01)
params1[1].Wh:normal(0,0.01)
params1[1].bh:normal(0,0.01)
params2[1].W:normal(0,0.01)
params2[1].b:normal(0,0.01)

-- Get the gradients closure magically:
local df = grad(f)

-- Max input length to train on
local maxLength = 30

-- Word dictionary to train:
local wordVecSize = 100
local words = torch.FloatTensor(nTokens, wordVecSize):normal(0,0.01)

-- Train it
local lr = 1
for epoch = 1,10 do
   print('Training Epoch #'..epoch)
   for i = 1,trainData:size(1)-maxLength-1,maxLength do
      -- Next sample:
      local x = trainData:narrow(1,i,maxLength)
      local y = trainData[i+maxLength]

      -- Select word vectors
      local xv = words:index(1, x:long())

      -- Grads:
      local vars = {params1=params1, params2=params2, x=xv}
      local grads,loss,prediction = df(vars, y)

      -- Update params:
      for k,param in ipairs(params1[1]) do
         param:add(-lr, grads.params1[k])
      end
      for k,param in ipairs(params2[1]) do
         param:add(-lr, grads.params1[k])
      end

      -- Update vectors:
      for i = 1,x:size(1) do
         words[i]:add(-lr, grads.x[i])
      end
   end
end
