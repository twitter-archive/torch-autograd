local sequence = require 'autograd.model.common'.sequence
local hasCudnn, cudnn = pcall(require, 'cudnn')
hasCudnn = hasCudnn and cudnn
local functionalize = require('autograd.nnwrapper').functionalize
local cast = require('autograd.util').cast
if hasCudnn then
   cudnn = functionalize('cudnn')
end
local nn = functionalize('nn')

local function SpatialLayer(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local kernelSize = opt.kernelSize or 5
   local padding = opt.padding or math.ceil(kernelSize-1)/2
   local inputFeatures = opt.inputFeatures or 3
   local outputFeatures = opt.outputFeatures or 16
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local activations = opt.activations
   local pooling = opt.pooling or 1
   local inputStride = opt.inputStride or 1
   local cuda = opt.cuda or false

   -- Set up modules
   local SpatialConvolution = nn.SpatialConvolutionMM
   local SpatialMaxPooling = nn.SpatialMaxPooling
   if cuda and hasCudnn then
      SpatialConvolution = cudnn.SpatialConvolution
      SpatialMaxPooling = cudnn.SpatialMaxPooling
   end

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- Dropout
   --------------------------------------
   if dropoutProb > 0 then
      table.insert(layers, nn.SpatialDropout(dropoutProb) )
   end

   -- Convolution
   --------------------------------------
   local l,p = SpatialConvolution(inputFeatures, outputFeatures, kernelSize, kernelSize, inputStride, inputStride, padding, padding)
   table.insert(layers, l)
   table.insert(params, p)
   layer2params[#layers] = #params

   -- Batch normalization
   --------------------------------------
   if batchNormalization then
      local l,p = nn.SpatialBatchNormalization(outputFeatures)
      table.insert(layers, l)
      table.insert(params, p)
      layer2params[#layers] = #params
   end

   -- Activations
   --------------------------------------
   if opt.activations then
      local activation
      if hasCudnn and cuda then
         activation = cudnn[activations]()
      else
         activation = nn[activations]()
      end
      table.insert(layers, activation)
   end

   -- Pooling
   --------------------------------------
   if pooling > 1 then
      table.insert(layers, SpatialMaxPooling(pooling, pooling))
   end

   return sequence(layers, layer2params), params, layers
end

return function(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local kernelSize = opt.kernelSize or 5
   local padding = opt.padding
   local inputFeatures = opt.inputFeatures or 3
   local hiddenFeatures = opt.hiddenFeatures or {16,32,64}
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local dropoutProbs = opt.dropoutProbs or {}
   local activations = opt.activations or 'ReLU'
   local poolings = opt.poolings or {1,1,1}
   local inputStride = opt.inputStride or 1
   local cuda = opt.cuda or false

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- add layers:
   for i,hiddens in ipairs(hiddenFeatures) do
      SpatialLayer({
         inputStride = inputStride,
         inputFeatures = inputFeatures,
         outputFeatures = hiddens,
         pooling = poolings[i],
         dropoutProb = dropoutProbs[i] or dropoutProb,
         activations = activations,
         batchNormalization = batchNormalization,
         kernelSize = kernelSize,
         padding = padding,
         cuda = cuda,
      }, params, layers, layer2params)
      inputFeatures = hiddens
      inputStride = 1
   end

   -- Type cast, if CUDA
   --------------------------------------
   if cuda then
      params = cast(params, "cuda")
   end


   -- layers
   return sequence(layers, layer2params), params, layers
end
