local sequence = require 'autograd.model.common'.sequence

local nn = require('autograd.main').nn

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

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- stack modules:
   if dropoutProb > 0 then
      table.insert(layers, nn.SpatialDropout(dropoutProb) )
   end
   table.insert(layers, nn.SpatialConvolutionMM(inputFeatures, outputFeatures, kernelSize, kernelSize, inputStride, inputStride, padding, padding) )
   do
      table.insert(params, {
         W = torch.zeros(outputFeatures, inputFeatures*kernelSize*kernelSize),
         b = torch.zeros(outputFeatures),
      })
      layer2params[#layers] = #params
   end
   if batchNormalization then
      table.insert(layers, nn.SpatialBatchNormalization(outputFeatures) )
      do
         table.insert(params, {
            W = torch.zeros(outputFeatures),
            b = torch.zeros(outputFeatures),
         })
         layer2params[#layers] = #params
      end
   end
   if opt.activations then
      table.insert(layers, nn[activations]())
   end
   if pooling > 1 then
      table.insert(layers, nn.SpatialMaxPooling(pooling, pooling) )
   end

   -- layers
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

   -- layers
   return sequence(layers, layer2params), params, layers
end
