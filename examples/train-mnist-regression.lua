-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'
local optim = require 'optim'

-- Helpers:
local function logMultiNomialLoss(out, target) return -torch.sum(torch.cmul(out,target)) end
local function logsumexp(array) return torch.log(torch.sum(torch.exp(array))) end
local function logsoftmax(array) return array - logsumexp(array) end
local function uniform(min, max, h, w) return torch.mul(torch.FloatTensor():rand(h,w), (max-min)) + min end
local function sigmoid(x) return 1.0/(torch.exp(-x)) end

-- Load in MNIST
local trainData, testData, classes = require('./setup-data.lua')()
trainData.y = util.oneHot(trainData.y)
testData.y = util.oneHot(testData.y)
local inputSize = trainData.x[1]:nElement()
local confusionMatrix = optim.ConfusionMatrix(classes)

-- Define our neural net
function neuralNet(params, input, target, return_prediction)
   local W2 = input * params.W[1] + params.B[1]
   local out = logsoftmax(W2)
   if return_prediction then
      return out
   else
      return logMultiNomialLoss(out, target)
   end
end

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
W1 = uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes), inputSize, #classes)
B1 = torch.FloatTensor(#classes):fill(0)

-- Trainable parameters:
local params = {
   W = {W1},
   B = {B1},
}

-- Train a neural network
for epoch=1,100 do
   print('EPOCH #'..epoch)
   for i=1,trainData.size do
      local x = trainData.x[i]:view(1,inputSize)
      local y = torch.view(trainData.y[i], 1, 10):clone()

      local dneuralNet = grad(neuralNet, 1, false)
      grads = dneuralNet(params,x,y)
      prediction = neuralNet(params,x,y,true)

      -- Update weights and biases
      for i=1,#params.W do
         params.W[i] = params.W[i] - grads.W[i] * 0.01
         params.B[i] = params.B[i] - grads.B[i] * 0.01
      end

      confusionMatrix:add(prediction, y)
      if i % 1000 == 0 then
         print(params.B[1][1])
         print(torch.sum(grads.W[1]))
         print(confusionMatrix)
         confusionMatrix:zero()
      end
   end
end
