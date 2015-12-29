local sequence = require 'autograd.model.common'.sequence
local NeuralNetwork = require 'autograd.model.NeuralNetwork'
local SpatialNetwork = require 'autograd.model.SpatialNetwork'


-- The ImageNet architecture for AlexNet would be set with the following options:
-- http://arxiv.org/pdf/1404.5997v2.pdf
-- https://github.com/eladhoffer/ImageNet-Training/blob/2cd055056082c05f7a7f5392fb7897c706cdb38a/Models/AlexNet_BN.lua
-- convOpt = {
   -- kernelSize = 3, -- NOTE: currently don't support per-layer kernel sizes. If we did, it'd be {11,5,3,3,3}
   -- hiddenFeatures = {64,192,384,256,256},
   -- batchNormalization = true,
   -- padding = 2, -- NOTE: currently don't support per-layer paddings. If we did, it'd be {2,2,1,1,1}
   -- dropoutProb = 0,
   -- activations = 'ReLU',
   -- inputStride = 1, -- NOTE: currently don't supported per-layer inputStrides. If we did, it'd be {4,1,1,1,1}
   -- poolings = {3,3,3,3,3} -- We don't set kW/H and dW/H separately.
-- }
mlpOpt = {
   hiddenFeatures = {4096,4096,1000},
   batchNormalization = true,
   dropoutProbs = {0.5,0.5,0},
   classifier = true,
   activations = "ReLU",
}

return function(imageDepth, imageHeight, imageWidth, convOpt, mlpOpt, params, layers, layer2params)

   -- Convolution options
   --------------------------------------------
   convOpt = convOpt or {}
   if convOpt.inputFeatures then
      print("Input features set will be overridden with the imageDepth value provided: " .. tostring(imageDepth))
   end
   convOpt.inputFeatures = imageDepth
   convOpt.kernelSize = convOpt.kernelSize or 5
   if convOpt.kernelSizes then
      error("Currently, per-layer kernel sizes are not supported")
   end
   convOpt.padding = convOpt.padding
   if convOpt.paddings then
      error("Currently, per-layer paddings are not supported")
   end
   convOpt.hiddenFeatures = convOpt.hiddenFeatures or {16,32,64}
   convOpt.batchNormalization = convOpt.batchNormalization or false
   convOpt.dropoutProb = convOpt.dropoutProb or 0
   convOpt.dropoutProbs = convOpt.dropoutProbs or {}
   convOpt.activations = convOpt.activations or 'ReLU'
   convOpt.poolings = convOpt.poolings or {1,1,1}
   convOpt.inputStride = convOpt.inputStride or 1
   convOpt.cuda = convOpt.cuda or false

   -- MLP options
   --------------------------------------------
   mlpOpt = mlpOpt or {}
   if mlpOpt.inputFeatures then
      error("Input features on the fully-connected layers cannot be manually set, do not specify")
   end
   mlpOpt.hiddenFeatures = mlpOpt.hiddenFeatures or {100,2}
   mlpOpt.batchNormalization = mlpOpt.batchNormalization or false
   mlpOpt.dropoutProb = mlpOpt.dropoutProb or 0
   mlpOpt.dropoutProbs = mlpOpt.dropoutProbs or {}
   mlpOpt.activations = mlpOpt.activations or 'ReLU'
   mlpOpt.classifier = mlpOpt.classifier or true -- classifier by default.
   mlpOpt.cuda = mlpOpt.cuda or false
   mlpOpt = mlpOpt or {}

   if (mlpOpt.cuda and not convOpt.cuda) or (not mlpOpt.cuda and convOpt.cuda) then
      print("")
      print("CUDA set on one, but not both of spatial and fully-connected layers. Setting all to CUDA.")
      mlpOpt.cuda = true
      convOpt.cuda = true
   end

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- Build convnet layers
   local sp,params,layers = SpatialNetwork(convOpt, params, layers, layer2params)

   -- Figure out convolution net output size (dependent on image size)
   local testInput = torch.randn(1, imageDepth, imageHeight, imageWidth):typeAs(params[1][1]):contiguous()
   local res = sp(params, testInput)
   mlpOpt.inputFeatures = res:size(2)*res:size(3)*res:size(4)

   -- Set up fully-connected layers to accept convolutional layer output
   local fn,params,layers = NeuralNetwork(mlpOpt, params, layers, layer2params)

   return sequence(layers, layer2params), params, layers

end

