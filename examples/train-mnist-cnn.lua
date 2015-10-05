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
-- for CNNs, we rely on efficient nn-provided primitives:
local reshape = grad.nn.Reshape(1,32,32)

local conv1 = grad.nn.SpatialConvolutionMM(1, 16, 5, 5)
local acts1 = grad.nn.Tanh()
local pool1 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

local conv2 = grad.nn.SpatialConvolutionMM(16, 16, 5, 5)
local acts2 = grad.nn.Tanh()
local pool2 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

local flatten = grad.nn.Reshape(16*5*5)
local linear = grad.nn.Linear(16*5*5, 10)

-- nn version:
function predict(params, input, target)
   local h1 = pool1(acts1(conv1(reshape(input), params.W[1], params.B[1])))
   local h2 = pool2(acts2(conv2(h1, params.W[2], params.B[2])))
   local h3 = linear(flatten(h2), params.W[3], params.B[3])
   local out = util.logSoftMax(h3)
   return out
end

function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = util.logMultiNomialLoss(prediction, target)
   return loss, prediction
end


-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
local W1 = torch.FloatTensor(16,1*5*5):uniform(-1/math.sqrt(16),1/math.sqrt(16))
local B1 = torch.FloatTensor(16):fill(0)
local W2 = torch.FloatTensor(16,16*5*5):uniform(-1/math.sqrt(16),1/math.sqrt(16))
local B2 = torch.FloatTensor(16):fill(0)
local W3 = torch.FloatTensor(#classes,16*5*5):uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes))
local B3 = torch.FloatTensor(#classes):fill(0)

-- Trainable parameters:
params = {
   W = {W1, W2, W3},
   B = {B1, B2, B3},
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
      local grads, loss, prediction = df(params,x,y)

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
