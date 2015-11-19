-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'

grad.optimize(true)

-- Load in MNIST
local trainData, testData, classes = require('./get-mnist.lua')()
local inputSize = trainData.x[1]:nElement()
local confusionMatrix = optim.ConfusionMatrix(classes)

-- What model to train:
local predict,f,params

-- Define our neural net
function predict(params, input, target)
   local h1 = torch.tanh(input * params.W[1] + params.B[1])
   local h2 = torch.tanh(h1 * params.W[2] + params.B[2])
   local h3 = h2 * params.W[3] + params.B[3]
   local out = util.logSoftMax(h3)
   return out
end

-- Define our training loss
function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = lossFuns.logMultinomialLoss(prediction, target)
   return loss, prediction
end

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
local W1 = torch.FloatTensor(inputSize,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
local B1 = torch.FloatTensor(50):fill(0)
local W2 = torch.FloatTensor(50,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
local B2 = torch.FloatTensor(50):fill(0)
local W3 = torch.FloatTensor(50,#classes):uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes))
local B3 = torch.FloatTensor(#classes):fill(0)

-- Trainable parameters:
params = {
   W = {W1, W2, W3},
   B = {B1, B2, B3},
}

-- Get the gradients closure magically:
local df = grad(f, { optimize = true })

-- Train a neural network
for epoch = 1,100 do
   print('Training Epoch #'..epoch)
   for i = 1,trainData.size do
      -- Next sample:
      local x = trainData.x[i]:view(1,inputSize)
      local y = torch.view(trainData.y[i], 1, 10)

      -- Grads:
      local grads, loss, prediction = df(params, x, y)

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
