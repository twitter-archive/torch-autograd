-- standard models
local model = {}

-- nn modules:
local nn = require('autograd.nnfuncwrapper')('nn')

-- util
local util = require 'autograd.util'
local node = require 'autograd.node'
local getValue = node.getValue

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

function model.SpatialNetwork(opt, params, layers, layer2params)
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
      model.SpatialLayer({
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

function model.RecurrentNetwork(opt, params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 10
   local hiddenFeatures = opt.hiddenFeatures or 100
   local outputType = opt.outputType or 'last' -- 'last' or 'all'

   -- TODO:
   --> move hidden states to a tensor once __newindex and __index
   --  are available in autograd, so that "all" mode returns a
   --  tensor instead of a table

   -- container:
   params = params or {}

   -- parameters:
   local p = {
      Wx = torch.zeros(hiddenFeatures, inputFeatures),
      bx = torch.zeros(hiddenFeatures),
      Wh = torch.zeros(hiddenFeatures, hiddenFeatures),
      bh = torch.zeros(hiddenFeatures),
   }
   table.insert(params, p)

   -- function:
   local f = function(params, x)
      -- dims:
      local p = params[1] or params
      local steps = getValue(x):size(1)

      -- hiddens:
      local hs = {}

      -- go over time:
      for t = 1,steps do
         local hx = p.Wx * torch.select(x,1,t) + p.bx
         local hh
         if t > 1 then
            hh = p.Wh * hs[t-1] + p.bh
         else
            hh = p.bh
         end
         hs[t] = torch.tanh(hx+hh)
      end

      -- output:
      if outputType == 'last' then
         -- return last hidden code:
         return hs[#hs]
      else
         -- return all:
         return hs
      end
   end

   -- layers
   return f, params
end

function model.RecurrentLSTMNetwork(opt, params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 10
   local hiddenFeatures = opt.hiddenFeatures or 100
   local outputType = opt.outputType or 'last' -- 'last' or 'all'

   -- TODO:
   --> move hidden states to a tensor once __newindex and __index
   --  are available in autograd, so that "all" mode returns a
   --  tensor instead of a table

   -- container:
   params = params or {}

   -- parameters:
   local p = {
      Wx = torch.zeros(4 * hiddenFeatures, inputFeatures),
      bx = torch.zeros(4 * hiddenFeatures),
      Wh = torch.zeros(4 * hiddenFeatures, hiddenFeatures),
      bh = torch.zeros(4 * hiddenFeatures),
   }
   table.insert(params, p)

   -- function:
   local f = function(params, x)
      -- dims:
      local p = params[1] or params
      local steps = getValue(x):size(1)

      -- hiddens:
      local hs = {}
      local cs = {}

      -- go over time:
      for t = 1,steps do
         -- pack all dot products:
         local hx = p.Wx * torch.select(x,1,t) + p.bx
         local hh
         if t > 1 then
            hh = p.Wh * hs[t-1] + p.bh
         else
            hh = p.bh
         end
         local sums = torch.view(hx+hh, 4, hiddenFeatures)

         -- batch compute gates:
         local sigmoids = util.sigmoid( torch.narrow(sums, 1,1,3) )
         local inputGate = torch.select(sigmoids, 1,1)
         local forgetGate = torch.select(sigmoids, 1,2)
         local outputGate = torch.select(sigmoids, 1,3)

         -- write inputs
         local tanhs = torch.tanh( torch.narrow(sums, 1,4,1) )
         local inputValue = torch.select(tanhs, 1,1)

         -- partial gatings:
         if t > 1 then
            cs[t] = torch.cmul(forgetGate, cs[t-1]) + torch.cmul(inputGate, inputValue)
         else
            cs[t] = torch.cmul(inputGate, inputValue)
         end

         -- next h
         hs[t] = torch.cmul(outputGate, torch.tanh(cs[t]))
      end

      -- output:
      if outputType == 'last' then
         -- return last hidden code:
         return hs[#hs]
      else
         -- return all:
         return hs
      end
   end

   -- layers
   return f, params
end

return model
