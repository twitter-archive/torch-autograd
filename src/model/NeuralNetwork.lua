local sequence = require 'autograd.model.common'.sequence
local nn = require('autograd.main').nn
hasCudnn, cudnn = pcall(require, 'cudnn')

local function NeuralLayer(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 3
   local outputFeatures = opt.outputFeatures or 16
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local activations = opt.activations
   local cuda = opt.cuda or false

   -- container
   layers = layers or {}
   params = params or {}
   layer2params = layer2params or {}

   -- Dropout
   --------------------------------------
   if dropoutProb > 0 then
      table.insert(layers, nn.SpatialDropout(dropoutProb) )
   end

   -- Fully-connected layer
   --------------------------------------
   local l,p = nn.Linear(inputFeatures, outputFeatures)
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
      table.insert(layers, nn[activations]())
   end

   -- layers
   return sequence(layers, layer2params), params, layers
end

return function(opt, params, layers, layer2params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 10
   local hiddenFeatures = opt.hiddenFeatures or {100,2}
   local batchNormalization = opt.batchNormalization or false
   local dropoutProb = opt.dropoutProb or 0
   local dropoutProbs = opt.dropoutProbs or {}
   local activations = opt.activations or 'ReLU'
   local classifier = opt.classifier or false
   local cuda = opt.cuda or false

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
      NeuralLayer({
         inputFeatures = inputFeatures,
         outputFeatures = hiddens,
         dropoutProb = dropoutProbs[i] or dropoutProb,
         activations = activations,
         batchNormalization = batchNormalization,
         cuda = cuda,
      }, params, layers, layer2params)
      inputFeatures = hiddens
   end

   -- layers
   return sequence(layers, layer2params), params, layers
end
