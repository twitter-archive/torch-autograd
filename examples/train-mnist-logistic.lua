local opt = lapp [[
Train a classifier on MNIST.
]]

-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'
local optim = require 'optim'

-- Load in MNIST
local trainData, testData, classes = require('./get-mnist.lua')()
local inputSize = trainData.x[1]:nElement()
local confusionMatrix = optim.ConfusionMatrix(classes)

-- What model to train:
local f,params

-- Define our neural net
function f(params, input, target, predictionOnly)
   local h1 = input * params.W[1] + params.B[1]
   local out = util.logSoftMax(h1)
   if predictionOnly then
      return out
   else
      return util.logMultiNomialLoss(out, target)
   end
end

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
local W1 = torch.FloatTensor(inputSize,#classes):uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes))
local B1 = torch.FloatTensor(#classes):fill(0)

-- Trainable parameters:
params = {
   W = {W1},
   B = {B1},
}

-- Get the gradients closure magically:
local df = grad(f)

-- Train a neural network
for epoch = 1,100 do
   print('Training Epoch #'..epoch)
   for i = 1,trainData.size do
      -- Next sample:
      local x = trainData.x[i]:view(1,inputSize)
      local y = torch.view(trainData.y[i], 1, 10)

      -- Grads:
      local grads = df(params,x,y)
      local prediction = f(params,x,y,true)

      -- Update weights and biases
      for i=1,#params.W do
         params.W[i] = params.W[i] - grads.W[i] * 0.01
         params.B[i] = params.B[i] - grads.B[i] * 0.01
      end

      -- Log performance:
      confusionMatrix:add(prediction[1], y[1])
      if i % 1000 == 0 then
         print(confusionMatrix)
         confusionMatrix:zero()
      end
   end
end
