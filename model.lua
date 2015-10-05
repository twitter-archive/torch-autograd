-- standard models
local model = {}

-- nn modules:
local nn = require 'autograd.nnfuncwrapper'

-- generic generator, from sequential list of layers:
local function sequence(layers)
   return function(params, input)
      for i,layer in ipairs(layers) do
         input = layer(input, params[i].W, params[i].b)
      end
      return input
   end
end

function model.NeuralLayer(opt, params, layers)
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

   -- dropout:
   if dropoutProb > 0 then
      table.insert(layers, nn.Dropout(dropoutProb))
      table.insert(params, {})
   end

   -- linear layer:
   table.insert(layers, nn.Linear(inputFeatures, outputFeatures))
   table.insert(params, {
      W = torch.zeros(outputFeatures, inputFeatures),
      b = torch.zeros(outputFeatures),
   })

   -- batch normalization:
   if batchNormalization then
      table.insert(layers, nn.BatchNormalization(outputFeatures))
      table.insert(params, {})
   end

   -- activations:
   if opt.activations then
      table.insert(layers, nn[activations]())
      table.insert(params, {})
   end

   -- layers
   return sequence(layers),params,layers
end

function model.NeuralNetwork(opt, layers)
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

   -- always add a reshape to force input dim:
   table.insert(layers, nn.Reshape(inputFeatures))
   table.insert(params, {})

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
      }, params, layers)
      inputFeatures = hiddens
   end

   -- layers
   return sequence(layers),params,layers
end

return model
