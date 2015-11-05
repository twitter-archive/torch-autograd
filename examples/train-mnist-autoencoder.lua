-- A comparison between autograd and nngraph
-- using an L2-regularized autoencoder with tied weights.

-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'
local optim = require 'optim'

-- -- Load in MNIST
-- local trainData, testData, classes = require('./get-mnist.lua')()
-- local inputSize = trainData.x[1]:nElement()
-- local confusionMatrix = optim.ConfusionMatrix(classes)

-- What model to train:
local predict,f,params

-- Define our neural net
function predict(params, input)
   -- Encoder
   local h1 = util.sigmoid(input * params.W[1] + params.B[1]:expand(input:size(1), params.B[1]:size(2)))
   local h2 = util.sigmoid(h1 * params.W[2] + params.B[2]:expand(input:size(1), params.B[2]:size(2)))
   local h3 = util.sigmoid(h2 * params.W[3] + params.B[3]:expand(input:size(1), params.B[3]:size(2)))
   
   -- Decoder
   local h4 = util.sigmoid(h3 * params.W[3]:t() + params.B[4]:expand(input:size(1), params.B[4]:size(2)))
   local h5 = util.sigmoid(h2 * params.W[2]:t() + params.B[5]:expand(input:size(1), params.B[5]:size(2)))
   local out = util.sigmoid(h1 * params.W[1]:t() + params.B[6]:expand(input:size(1), params.B[6]:size(2)))

   return out
end

-- Define our training loss
function f(params, input, target, l2Lambda)
   -- Reconstruction loss
   local prediction = predict(params, input)
   local loss = util.logBCELoss(prediction, input)
   print(prediction)
   print(1-prediction)
   -- L2 penalty on the weights
   for i=1,#params.W do
      loss = loss + l2Lambda * torch.sum(torch.pow(params.W[i],2))
   end

   return loss, prediction
end

sizes = {}
sizes['input'] = 5
sizes['h1'] = 7
sizes['h2'] = 3
sizes['h3'] = 10

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
local W1 = torch.FloatTensor(sizes['input'],sizes['h1']):uniform(-1/math.sqrt(sizes['h1']),1/math.sqrt(sizes['h1']))
local W2 = torch.FloatTensor(sizes['h1'],sizes['h2']):uniform(-1/math.sqrt(sizes['h2']),1/math.sqrt(sizes['h2']))
local W3 = torch.FloatTensor(sizes['h2'],sizes['h3']):uniform(-1/math.sqrt(sizes['h3']),1/math.sqrt(sizes['h3']))
local B1 = torch.FloatTensor(1, sizes['h1']):fill(0)
local B2 = torch.FloatTensor(1, sizes['h2']):fill(0)
local B3 = torch.FloatTensor(1, sizes['h3']):fill(0)
local B4 = torch.FloatTensor(1, sizes['h2']):fill(0)
local B5 = torch.FloatTensor(1, sizes['h1']):fill(0)
local B6 = torch.FloatTensor(1, sizes['input']):fill(0)

-- Trainable parameters:
params = {
   W = {W1, W2, W3},
   B = {B1, B2, B3, B4, B5, B6},
}

x = torch.randn(3, 5):float()
print(f(params, x, 2.1))

-- -- Define our network
-- function predict(params, input, target)
--    local h1 = pool1(acts1(conv1(reshape(input), params.W[1], params.B[1])))
--    local h2 = pool2(acts2(conv2(h1, params.W[2], params.B[2])))
--    local h3 = linear(flatten(h2), params.W[3], params.B[3])
--    local out = util.logSoftMax(h3)
--    return out
-- end

-- -- Define our loss function
-- function f(params, input, target)
--    local prediction = predict(params, input, target)
--    local loss = util.logMultinomialLoss(prediction, target)
--    return loss, prediction
-- end


-- -- Define our parameters
-- -- [-1/sqrt(#output), 1/sqrt(#output)]
-- torch.manualSeed(0)
-- local W1 = torch.FloatTensor(16,1*5*5):uniform(-1/math.sqrt(16),1/math.sqrt(16))
-- local B1 = torch.FloatTensor(16):fill(0)
-- local W2 = torch.FloatTensor(16,16*5*5):uniform(-1/math.sqrt(16),1/math.sqrt(16))
-- local B2 = torch.FloatTensor(16):fill(0)
-- local W3 = torch.FloatTensor(#classes,16*5*5):uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes))
-- local B3 = torch.FloatTensor(#classes):fill(0)

-- -- Trainable parameters:
-- params = {
--    W = {W1, W2, W3},
--    B = {B1, B2, B3},
-- }

-- -- Get the gradients closure magically:
-- local df = grad(f)

-- -- Train a neural network
-- for epoch = 1,100 do
--    print('Training Epoch #'..epoch)
--    for i = 1,trainData.size do
--       -- Next sample:
--       local x = trainData.x[i]:view(1,inputSize)
--       local y = torch.view(trainData.y[i], 1, 10)

--       -- Grads:
--       local grads, loss, prediction = df(params,x,y)

--       -- Update weights and biases
--       for i=1,#params.W do
--          params.W[i] = params.W[i] - grads.W[i] * 0.01
--          params.B[i] = params.B[i] - grads.B[i] * 0.01
--       end

--       -- Log performance:
--       confusionMatrix:add(prediction[1], y[1])
--       if i % 1000 == 0 then
--          print(confusionMatrix)
--          confusionMatrix:zero()
--       end
--    end
-- end
