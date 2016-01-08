-- A comparison between autograd and nngraph
-- using an L2-regularized autoencoder with tied weights.

-- Libs
local grad = require 'autograd'
local lossFuns = require 'autograd.loss'
local util = require 'autograd.util'
local Value = require 'autograd.runtime.codegen.Value'
local gradcheck = require 'autograd.gradcheck'
local optim = require 'optim'

grad.optimize(true)

-- Load in MNIST
local trainData, testData, classes = require('./get-mnist.lua')()
trainData.x = trainData.x:view(trainData.x:size(1), -1):double()
local inputSize = trainData.x[1]:nElement()

-- What model to train:
local predict,f,params

-- Define our neural net
function predict(params, input)
   -- Encoder
   local h1 = util.sigmoid(input * params.W[1] + torch.expand(params.B[1], torch.size(input, 1), torch.size(params.B[1], 2)))
   local h2 = util.sigmoid(h1 * params.W[2] + torch.expand(params.B[2], torch.size(input, 1), torch.size(params.B[2], 2)))
   local h3 = util.sigmoid(h2 * params.W[3] + torch.expand(params.B[3], torch.size(input, 1), torch.size(params.B[3], 2)))
   -- Decoder
   local h4 = util.sigmoid(h3 * torch.t(params.W[3]) + torch.expand(params.B[4], torch.size(input, 1), torch.size(params.B[4], 2)))
   local h5 = util.sigmoid(h4 * torch.t(params.W[2]) + torch.expand(params.B[5], torch.size(input, 1), torch.size(params.B[5], 2)))
   local out = util.sigmoid(h5 * torch.t(params.W[1]) + torch.expand(params.B[6], torch.size(input, 1), torch.size(params.B[6], 2)))

   return out
end

-- Define our training loss
function f(params, input, l2Lambda)
   -- Reconstruction loss
   local prediction = predict(params, input)
   local loss = lossFuns.logBCELoss(prediction, input, 1e-6) / torch.size(input, 1)
   -- L2 penalty on the weights
   for i=1,Value.len(params.W) do
      loss = loss + l2Lambda * torch.sum(torch.pow(params.W[i],2))
   end

   return loss, prediction
end

-- Get the gradients closure magically:
local df = grad(f, { optimize = true })

sizes = {}
sizes['input'] = inputSize
sizes['h1'] = 50
sizes['h2'] = 25
sizes['h3'] = 10

-- L2 penalty strength
l2Lambda = 0.0

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
local W1 = torch.DoubleTensor(sizes['input'],sizes['h1']):uniform(-1/math.sqrt(sizes['h1']),1/math.sqrt(sizes['h1']))
local W2 = torch.DoubleTensor(sizes['h1'],sizes['h2']):uniform(-1/math.sqrt(sizes['h2']),1/math.sqrt(sizes['h2']))
local W3 = torch.DoubleTensor(sizes['h2'],sizes['h3']):uniform(-1/math.sqrt(sizes['h3']),1/math.sqrt(sizes['h3']))
local B1 = torch.DoubleTensor(1, sizes['h1']):fill(0)
local B2 = torch.DoubleTensor(1, sizes['h2']):fill(0)
local B3 = torch.DoubleTensor(1, sizes['h3']):fill(0)
local B4 = torch.DoubleTensor(1, sizes['h2']):fill(0)
local B5 = torch.DoubleTensor(1, sizes['h1']):fill(0)
local B6 = torch.DoubleTensor(1, sizes['input']):fill(0)

-- Trainable parameters:
params = {
   W = {W1, W2, W3},
   B = {B1, B2, B3, B4, B5, B6},
}

-- Train a neural network
for epoch = 1,100 do
   print('Training Epoch #'..epoch)
   for i = 1,trainData.size / 1000 do
      -- Next minibatch:
      local x = trainData.x[{{(i-1) * 100 + 1, i * 100}, {}}]

      -- Grads:
      local grads, loss, prediction = df(params,x,l2Lambda)

      -- Update weights and biases
      for i=1,#params.W do
         params.W[i] = params.W[i] - grads.W[i] * 0.01
      end

      for i=1,#params.B do
         params.B[i] = params.B[i] - grads.B[i] * 0.01
      end

   end

   -- Log performance:
   print('Cross-entropy loss: '..f(params, trainData.x[{{1,10000}, {}}], l2Lambda))
end
