-- Options
local opt = lapp [[
Train an LSTM to fit the Penn Treebank dataset.

Options:
   --nEpochs        (default 5)    nb of epochs
   --bpropLength    (default 20)   max backprop steps
   --wordDim        (default 200)  word vector dimensionality
   --hiddens        (default 200)  nb of hidden units
   --capEpoch       (default -1)   cap epoch to given number of steps (for debugging)
   --reportEvery    (default 100)  report training accuracy every N steps
   --learningRate   (default 1)    learning rate
   --clipGrads      (default 5)    clip gradients
   --paramRange     (default .1)   initial parameter range
   --cuda                          run on CUDA device
]]

-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'
local getValue = require 'autograd.node'.getValue

-- CUDA?
if opt.cuda then
   require 'cutorch'
end

-- Load in PENN Treebank dataset
local trainData, valData, testData, dict = require('./get-penn.lua')()
local nTokens = #dict.id2word

-- Max input length to train on
local maxLength = opt.bpropLength

print('Loaded datasets: ', {
   train = trainData,
   validation = valData,
   test = testData,
   nTokens = nTokens,
})

-- Define LSTM layers:
local lstm,params = grad.model.RecurrentLSTMNetwork({
   inputFeatures = opt.wordDim,
   hiddenFeatures = opt.hiddens,
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

-- Linear classifier params:
params[2] = {
   W = torch.Tensor(#dict.id2word, opt.hiddens),
   b = torch.Tensor(#dict.id2word),
}

-- Init weights + cast:
for i,weights in ipairs(params) do
   for k,weight in pairs(weights) do
      if opt.cuda then
         weights[k] = weights[k]:cuda()
      else
         weights[k] = weights[k]:float()
      end
      weights[k]:uniform(-opt.paramRange, opt.paramRange)
   end
end

-- Get the gradients closure magically:
local df = grad(f)

-- Word dictionary to train:
local words
if opt.cuda then
   words = torch.CudaTensor(nTokens, opt.wordDim):uniform(-opt.paramRange, opt.paramRange)
else
   words = torch.FloatTensor(nTokens, opt.wordDim):uniform(-opt.paramRange, opt.paramRange)
end

-- Epoch length
local epochLength = trainData:size(1)
if tonumber(opt.capEpoch) > 0 then
   epochLength = opt.capEpoch
end

-- Train it
local lr = opt.learningRate
local reportEvery = opt.reportEvery
local valPerplexity = math.huge
local istart = 1
for epoch = 1,opt.nEpochs do
   -- For debugging mostly - if epoch size is smaller than
   -- training data size, then start from random offset
   if epochLength < trainData:size(1) then
      istart = torch.random(1,trainData:size(1)-epochLength+1)
      print('\nSetting epoch size to ' .. epochLength .. ', starting at offset = ' .. istart)
   end

   -- Train:
   print('\nTraining Epoch #'..epoch)
   local aloss = 0
   local maxGrad = 0
   local lstmState -- clear LSTM state at each new epoch
   for i = 1,epochLength-maxLength,maxLength do
      xlua.progress(i,epochLength)

      -- Next sequence:
      local x = trainData:narrow(1,i,maxLength)
      local y = trainData:narrow(1,i+1,maxLength)

      -- Select word vectors
      local xv = words:index(1, x:long())

      -- CUDA?
      if opt.cuda then
         xv = xv:cuda()
         y = y:cuda()
      end

      -- Grads:
      local grads,loss,newLstmState = df({params=params, x=xv}, y, lstmState)

      -- Preserve state for next iteration
      lstmState = newLstmState

      -- Update params:
      for i,params in ipairs(params) do
         for k,param in pairs(params) do
            local g = grads.params[i][k]
            g:clamp(-opt.clipGrads, opt.clipGrads)
            param:add(-lr, g)
         end
      end

      -- Update vectors:
      for i = 1,x:size(1) do
         local g = grads.x[i]
         words[i]:add(-lr, g)
      end

      -- Loss: exponentiate nll gives perplexity
      aloss = aloss + loss
      if ((i-1)/maxLength+1) % reportEvery == 0 then
         aloss = aloss / reportEvery
         local perplexity = math.exp(aloss)
         print('\nAverage training perplexity = ' .. perplexity)
         aloss = 0
      end

      -- TODO: get rid of this once autograd allocates less
      collectgarbage()
   end

   -- Validate:
   print('\nValidation #'..epoch)
   local aloss = 0
   local steps = 0
   local lstmState -- clear LSTM state at each new epoch
   for i = 1,valData:size(1)-maxLength,maxLength do
      -- Progress:
      xlua.progress(i,valData:size(1))

      -- Next sequence:
      local x = valData:narrow(1,i,maxLength)
      local y = valData:narrow(1,i+1,maxLength)

      -- Select word vectors
      local xv = words:index(1, x:long())

      -- CUDA?
      if opt.cuda then
         xv = xv:cuda()
         y = y:cuda()
      end

      -- Estimate loss:
      local loss,newLstmState = f({params=params, x=xv}, y, lstmState)

      -- Preserve state for next iteration
      lstmState = newLstmState

      -- Loss: exponentiate nll gives perplexity
      aloss = aloss + loss
      steps = steps + 1
   end
   aloss = aloss / steps
   local perplexity = math.exp(aloss)
   print('\nValidation perplexity = ' .. perplexity)

   -- Learning rate scheme:
   if perplexity < valPerplexity then
      -- Progress made!
      valPerplexity = perplexity
   else
      -- No progress made, decrease learning rate
      lr = lr / 2
      print('Validation perplexity regressed, decreasing learning rate to: ' .. lr)
   end
end

-- Test:
print('\n\nTest set performance...:')
local aloss = 0
local steps = 0
local lstmState -- clear LSTM state at each new epoch
for i = 1,testData:size(1)-maxLength,maxLength do
   -- Progress:
   xlua.progress(i,testData:size(1))

   -- Next sequence:
   local x = testData:narrow(1,i,maxLength)
   local y = testData:narrow(1,i+1,maxLength)

   -- Select word vectors
   local xv = words:index(1, x:long())

   -- CUDA?
   if opt.cuda then
      xv = xv:cuda()
      y = y:cuda()
   end

   -- Estimate loss:
   local loss,newLstmState = f({params=params, x=xv}, y, lstmState)

   -- Preserve state for next iteration
   lstmState = newLstmState

   -- Loss: exponentiate nll gives perplexity
   aloss = aloss + loss
   steps = steps + 1
end
aloss = aloss / steps
local perplexity = math.exp(aloss)
print('\nTest set perplexity = ' .. perplexity)
