-- standard models
local model = {}

-- nn modules:
local nn = require 'autograd.nnfuncwrapper'

-- generic generator, from sequential list of layers:
local function sequence(layers, layer2params)
   return function(params, input)
      for i,layer in ipairs(layers) do
         local paramsi = layer2params[i]
         if paramsi then
            input = layer(input, params[paramsi].W, params[paramsi].b)
         else
            input = layer(input)
         end
      end
      return input
   end
end

function model.NeuralLayer(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 3
   local outputFeatures = opt.outputFeatures or 16
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local activations = opt.activations
   local channelWise = opt.channelWise

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- dropout:
   if dropoutProb > 0 then
      table.insert(layers, nn.Dropout(dropoutProb))
   end

   -- linear layer:
   table.insert(layers, nn.Linear(inputFeatures, outputFeatures))
   table.insert(params, {
      W = torch.zeros(outputFeatures, inputFeatures),
      b = torch.zeros(outputFeatures),
   })
   layer2params[#layers] = #params

   -- batch normalization:
   if batchNormalization then
      table.insert(layers, nn.BatchNormalization(outputFeatures))
      table.insert(params, {
         W = torch.zeros(outputFeatures),
         b = torch.zeros(outputFeatures),
      })
      layer2params[#layers] = #params
   end

   -- activations:
   if opt.activations then
      table.insert(layers, nn[activations]())
   end

   -- layers
   return sequence(layers, layer2params), params, layers
end

function model.NeuralNetwork(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 10
   local hiddenFeatures = opt.hiddenFeatures or {100,2}
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local dropoutProbs = opt.dropoutProbs or {}
   local activations = opt.activations or 'ReLU'
   local channelWise = opt.channelWise
   local classifier = opt.classifier or false

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- always add a reshape to force input dim:
   table.insert(layers, nn.Reshape(inputFeatures))

   -- add layers:
   for i,hiddens in ipairs(hiddenFeatures) do
      if classifier and i == #hiddenFeatures then
         activations = nil
         batchNormalization = nil
      end
      model.NeuralLayer({
         inputFeatures = inputFeatures,
         outputFeatures = hiddens,
         dropoutProb = dropoutProbs[i] or dropoutProb,
         activations = activations,
         channelWise = channelWise,
         batchNormalization = batchNormalization,
      }, params, layers, layer2params)
      inputFeatures = hiddens
   end

   -- layers
   return sequence(layers, layer2params), params, layers
end

function model.SpatialLayer(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local kernelSize = opt.kernelSize or 5
   local kernelSplit = opt.kernelSplit
   local padding = opt.padding or math.ceil(kernelSize-1)/2
   local inputFeatures = opt.inputFeatures or 3
   local outputFeatures = opt.outputFeatures or 16
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local activations = opt.activations
   local channelWise = opt.channelWise
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

function model.SpatialNetwork(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local kernelSize = opt.kernelSize or 5
   local kernelSplit = opt.kernelSplit
   local padding = opt.padding
   local inputFeatures = opt.inputFeatures or 3
   local hiddenFeatures = opt.hiddenFeatures or {16,32,64}
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local dropoutProbs = opt.dropoutProbs or {}
   local activations = opt.activations or 'ReLU'
   local channelWise = opt.channelWise
   local poolings = opt.poolings or {1,1,1}
   local inputStride = opt.inputStride or 1

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- add layers:
   for i,hiddens in ipairs(hiddenFeatures) do
      model.SpatialLayer({
         inputStride = inputStride,
         inputFeatures = inputFeatures,
         outputFeatures = hiddens,
         pooling = poolings[i],
         dropoutProb = dropoutProbs[i] or dropoutProb,
         activations = activations,
         channelWise = channelWise,
         batchNormalization = batchNormalization,
         kernelSize = kernelSize,
         kernelSplit = kernelSplit,
         padding = padding,
         cuda = cuda,
      }, params, layers, layer2params)
      inputFeatures = hiddens
      inputStride = 1
   end

   -- layers
   return sequence(layers, layer2params), params, layers
end

return model
