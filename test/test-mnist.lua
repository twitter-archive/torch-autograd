require 'autograd'
require 'util'
optim = require 'optim'

local function logMultiNomialLoss(out, target) return -torch.sum(torch.cmul(out,target)) end
local function logsumexp(array) return torch.log(torch.sum(torch.exp(array))) end
local function logsoftmax(array) return array - logsumexp(array) end
local function uniform(min, max, h, w) return torch.mul(torch.FloatTensor():rand(h,w), (max-min)) + min end

-- Load in MNIST
trainData, testData, classes = require('./setup-data.lua')()
trainData.y = oneHot(trainData.y)
testData.y = oneHot(testData.y)
inputSize = trainData.x[1]:nElement()
confusionMatrix = optim.ConfusionMatrix(classes)

-- Define our neural net
function neuralNet(params, input, target, return_prediction)
   local W2 = torch.tanh(input * params.W[1] + params.B[1])
   local W3 = torch.tanh(W2 * params.W[2] + params.B[2])
   local W4 = W3 * params.W[3] + params.B[3]
   local out = logsoftmax(W4)
   if return_prediction then
      return out
   else
      return logMultiNomialLoss(out, target)
   end
end

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
W1 = uniform(-1/math.sqrt(50),1/math.sqrt(50), inputSize, 50)
B1 = torch.FloatTensor(50):fill(0)
W2 = uniform(-1/math.sqrt(50),1/math.sqrt(50), 50, 50)
B2 = torch.FloatTensor(50):fill(0)
W3 = uniform(-1/math.sqrt(10),1/math.sqrt(10), 50, #classes)
B3 = torch.FloatTensor(#classes):fill(0)

local params = {
   W = {W1, W2, W3}, 
   B = {B1, B2, B3},
}


-- Train a neural network
for epoch=1,100 do
   print('Training Epoch #'..epoch)
   for i=1,trainData.size do
      local x = trainData.x[i]:view(1,inputSize)
      local y = torch.view(trainData.y[i], 1, 10):clone()

      local dneuralNet = grad(neuralNet)
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