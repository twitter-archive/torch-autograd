-- Options
local opt = lapp [[
Train an LSTM to fit the Penn Treebank dataset.

Options:
   --nEpochs        (default 5)       nb of epochs
   --bpropLength    (default 20)      max backprop steps
   --batchSize      (default 20)      batch size
   --wordDim        (default 200)     word vector dimensionality
   --hiddens        (default 200)     nb of hidden units
   --capEpoch       (default -1)      cap epoch to given number of steps (for debugging)
   --reportEvery    (default 100)     report training accuracy every N steps
   --learningRate   (default 1)       learning rate
   --maxGradNorm    (default 5)       cap gradient norm
   --paramRange     (default .1)      initial parameter range
   --dropout        (default 0)       dropout probability on hidden states
   --type           (default double)  tensor type: cuda | float | double
]]

-- Libs
local d = require 'autograd'
local util = require 'autograd.util'
local getValue = require 'autograd.node'.getValue
local model = require 'autograd.model'
local _ = require 'moses'

-- CUDA?
if opt.type == 'cuda' then
   require 'cutorch'
end

-- Load in PENN Treebank dataset
local trainData, valData, testData, dict = require('./get-penn.lua')()
local nTokens = #dict.id2word

-- Move data to CUDA
if opt.type == 'cuda' then
   trainData = trainData:cuda()
   testData = testData:cuda()
   valData = valData:cuda()
elseif opt.type == 'double' then
   trainData = trainData:double()
   testData = testData:double()
   valData = valData:double()
end

print('Loaded datasets: ', {
   train = trainData,
   validation = valData,
   test = testData,
   nTokens = nTokens,
})

-- Define LSTM layers:
local lstm1,params = model.RecurrentLSTMNetwork({
   inputFeatures = opt.wordDim,
   hiddenFeatures = opt.hiddens,
   outputType = 'all',
})
local lstm2 = model.RecurrentLSTMNetwork({
   inputFeatures = opt.hiddens,
   hiddenFeatures = opt.hiddens,
   outputType = 'all',
}, params)

-- Dropout
local dropout = function(state)
   local keep = 1 - opt.dropout
   if keep == 1 then return state end
   local sv = getValue(state)
   local keep = sv.new(sv:size()):bernoulli(keep):mul(1/keep)
   return torch.cmul(state, keep)
end
local bypass = function(state) return state end
local regularize = dropout

-- Shortcuts
local nElements = opt.batchSize*opt.bpropLength
local nClasses = #dict.id2word

-- Use built-in nn modules:
local lsm = d.nn.LogSoftMax()
local lossf = d.nn.ClassNLLCriterion()

-- Complete trainable function:
local f = function(inputs, y, prevState)
   -- N elements:
   local nElements = getValue(inputs.x):size(1) * getValue(inputs.x):size(2)

   -- Encode all inputs through LSTM layers:
   local h1,newState1 = lstm1(inputs.params[1], regularize(inputs.x), prevState[1])
   local h2,newState2 = lstm2(inputs.params[2], regularize(h1), prevState[2])

   -- Flatten batch + temporal
   local h2f = torch.view(h2, nElements, opt.hiddens)
   local yf = torch.view(y, nElements)

   -- Linear classifier:
   local h3 = regularize(h2f) * inputs.params[3].W + torch.expand(inputs.params[3].b, nElements, nClasses)

   -- Lsm
   local yhat = lsm(h3)

   -- Loss:
   local loss = lossf(yhat, yf)

   -- Equivalent to this (which is way slower for now):
   -- local loss = 0
   -- for i = 1,nElements do
   --    local yhati = torch.select(yhat,1,i)
   --    local yfi = torch.select(yf,1,i)
   --    loss = loss - torch.sum( torch.narrow(yhati, 1, yfi, 1) )
   -- end
   -- loss = loss / nElements

   -- Return avergage loss
   return loss, {newState1, newState2}
end

-- Training eval
local trainf = function(...)
   regularize = dropout
   return f(...)
end

-- Test eval
local testf = function(...)
   regularize = bypass
   return f(...)
end

-- Linear classifier params:
table.insert(params, {
   W = torch.Tensor(opt.hiddens, #dict.id2word),
   b = torch.Tensor(1, #dict.id2word),
})

-- Init weights + cast:
for i,weights in ipairs(params) do
   for k,weight in pairs(weights) do
      if opt.type == 'cuda' then
         weights[k] = weights[k]:cuda()
      elseif opt.type == 'double' then
         weights[k] = weights[k]:double()
      else
         weights[k] = weights[k]:float()
      end
      weights[k]:uniform(-opt.paramRange, opt.paramRange)
   end
end

-- Word dictionary to train:
local words
if opt.type == 'cuda' then
   words = torch.CudaTensor(nTokens, opt.wordDim):uniform(-opt.paramRange, opt.paramRange)
elseif opt.type == 'double' then
   words = torch.DoubleTensor(nTokens, opt.wordDim):uniform(-opt.paramRange, opt.paramRange)
else
   words = torch.FloatTensor(nTokens, opt.wordDim):uniform(-opt.paramRange, opt.paramRange)
end

-- Reformat training data for batches:
local epochLength = math.floor(trainData:size(1) / opt.batchSize)
trainData = trainData:narrow(1,1,epochLength*opt.batchSize):view(opt.batchSize, epochLength)

-- Optional cap:
if tonumber(opt.capEpoch) > 0 then
   epochLength = opt.capEpoch
end

-- Train it
local lr = opt.learningRate
local reportEvery = opt.reportEvery
local valPerplexity = math.huge
for epoch = 1,opt.nEpochs do
   -- Train:
   print('\nTraining Epoch #'..epoch)
   local aloss = 0
   local maxGrad = 0
   local lstmState = {} -- clear LSTM state at each new epoch
   local grads,loss
   for i = 1,epochLength-opt.bpropLength,opt.bpropLength do
      xlua.progress(i,epochLength)

      -- Next sequence:
      local x = trainData:narrow(2,i,opt.bpropLength):contiguous()
      local y = trainData:narrow(2,i+1,opt.bpropLength):contiguous()

      -- Select word vectors
      local xv = words:index(1, x:view(-1):long()):view(opt.batchSize, opt.bpropLength, opt.wordDim)

      -- Grads:
      grads,loss,lstmState = d(trainf)({params=params, x=xv}, y, lstmState)

      -- Cap gradient norms:
      for i,grad in ipairs(_.flatten(grads)) do
         local norm = grad:norm()
         if norm > opt.maxGradNorm then
            grad:mul( opt.maxGradNorm / norm )
         end
      end

      -- Update params:
      for i,params in ipairs(params) do
         for k,param in pairs(params) do
            local g = grads.params[i][k]
            param:add(-lr, g)
         end
      end

      -- Update vectors:
      local gradx = grads.x:view(nElements, opt.wordDim)
      for i = 1,nElements do
         words[i]:add(-lr, gradx[i])
      end

      -- Loss: exponentiate nll gives perplexity
      aloss = aloss + loss
      if ((i-1)/opt.bpropLength+1) % reportEvery == 0 then
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
   local lstmState = {}
   local loss
   for i = 1,valData:size(1)-opt.bpropLength,opt.bpropLength do
      -- Progress:
      xlua.progress(i,valData:size(1))

      -- Next sequence:
      local x = valData:narrow(1,i,opt.bpropLength)
      local y = valData:narrow(1,i+1,opt.bpropLength)

      -- Select word vectors
      local xv = words:index(1, x:long())

      -- Reshape to batch of 1
      xv = xv:view(1,opt.bpropLength,opt.wordDim)
      y = y:view(1,opt.bpropLength)

      -- Estimate loss:
      loss,lstmState = testf({params=params, x=xv}, y, lstmState)

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
local lstmState = {}
local loss
for i = 1,testData:size(1)-opt.bpropLength,opt.bpropLength do
   -- Progress:
   xlua.progress(i,testData:size(1))

   -- Next sequence:
   local x = testData:narrow(1,i,opt.bpropLength)
   local y = testData:narrow(1,i+1,opt.bpropLength)

   -- Select word vectors
   local xv = words:index(1, x:long())

   -- Reshape to batch of 1
   xv = xv:view(1,opt.bpropLength,opt.wordDim)
   y = y:view(1,opt.bpropLength)

   -- Estimate loss:
   loss,lstmState = testf({params=params, x=xv}, y, lstmState)

   -- Loss: exponentiate nll gives perplexity
   aloss = aloss + loss
   steps = steps + 1
end
aloss = aloss / steps
local perplexity = math.exp(aloss)
print('\nTest set perplexity = ' .. perplexity)
