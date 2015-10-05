local opt = lapp [[
Train a classifier on MNIST.

Options:
   --model  (default mlp)  model can be: mlp | logistic | cnn
]]

-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'
local optim = require 'optim'

-- Helpers:
local function logMultiNomialLoss(out, target) return -torch.sum(torch.cmul(out,target)) end
local function logSumExp(array) return torch.log(torch.sum(torch.exp(array))) end
local function logSoftMax(array) return array - logSumExp(array) end

-- Load in MNIST
local trainData, testData, classes = require('./get-mnist.lua')()
trainData.y = util.oneHot(trainData.y)
testData.y = util.oneHot(testData.y)
local inputSize = trainData.x[1]:nElement()
local confusionMatrix = optim.ConfusionMatrix(classes)

-- What model to train:
local f,params
if opt.model == 'mlp' then
   -- Define our neural net
   function f(params, input, target, predictionOnly)
      local h1 = torch.tanh(input * params.W[1] + params.B[1])
      local h2 = torch.tanh(h1 * params.W[2] + params.B[2])
      local h3 = h2 * params.W[3] + params.B[3]
      local out = logSoftMax(h3)
      if predictionOnly then
         return out
      else
         return logMultiNomialLoss(out, target)
      end
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

elseif opt.model == 'logistic' then
   -- Define our neural net
   function f(params, input, target, predictionOnly)
      local h1 = input * params.W[1] + params.B[1]
      local out = logSoftMax(h1)
      if predictionOnly then
         return out
      else
         return logMultiNomialLoss(out, target)
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

elseif opt.model == 'cnn' then
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
   function f(params, input, target, predictionOnly)
      local h1 = pool1(acts1(conv1(reshape(input), params.W[1], params.B[1])))
      local h2 = pool2(acts2(conv2(h1, params.W[2], params.B[2])))
      local h3 = linear(flatten(h2), params.W[3], params.B[3])
      local out = logSoftMax(h3)
      if predictionOnly then
         return out
      else
         return logMultiNomialLoss(out, target)
      end
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
end

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
      grads = df(params,x,y)
      prediction = f(params,x,y,true)

      -- Update weights and biases
      for i=1,#params.W do
         params.W[i] = params.W[i] - grads.W[i] * 0.01
         params.B[i] = params.B[i] - grads.B[i] * 0.01
      end

      -- Log performance:
      confusionMatrix:add(prediction, y)
      if i % 1000 == 0 then
         print(confusionMatrix)
         confusionMatrix:zero()
      end
   end
end
