-- util
local util = require 'autograd.util'
local functionalize = require('autograd.nnwrapper').functionalize
local nn = functionalize('nn')

return function(opt, params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 10
   local hiddenFeatures = opt.hiddenFeatures or 100
   local outputType = opt.outputType or 'last' -- 'last' or 'all'
   local batchNormalization = opt.batchNormalization or false
   local maxBatchNormalizationLayers = opt.maxBatchNormalizationLayers or 10

   -- containers:
   params = params or {}
   local layers = {}

   -- parameters:
   local p = {
      W = torch.zeros(inputFeatures+hiddenFeatures, 4 * hiddenFeatures),
      b = torch.zeros(1, 4 * hiddenFeatures),
   }
   if batchNormalization then
     -- translation and scaling parameters are shared across time.
     local lstm_bn, p_lstm_bn = nn.BatchNormalization(4 * hiddenFeatures)
     local cell_bn, p_cell_bn = nn.BatchNormalization(hiddenFeatures)

     layers.lstm_bn = {lstm_bn}
     layers.cell_bn = {cell_bn}

     for i=2,maxBatchNormalizationLayers do
       local lstm_bn = nn.BatchNormalization(4 * hiddenFeatures)
       local cell_bn = nn.BatchNormalization(hiddenFeatures)
       layers.lstm_bn[i] = lstm_bn
       layers.cell_bn[i] = cell_bn
     end

     -- initializing scaling to < 1 is recommended for LSTM batch norm.
     p.lstm_bn_1 = p_lstm_bn[1]:fill(0.1)
     p.lstm_bn_2 = p_lstm_bn[2]:zero()
     p.cell_bn_1 = p_cell_bn[1]:fill(0.1)
     p.cell_bn_2 = p_cell_bn[2]:zero()
   end
   table.insert(params, p)

   -- function:
   local f = function(params, x, prevState, layers)
      -- dims:
      local p = params[1] or params
      if torch.nDimension(x) == 2 then
         x = torch.view(x, 1, torch.size(x, 1), torch.size(x, 2))
      end
      local batch = torch.size(x, 1)
      local steps = torch.size(x, 2)

      -- hiddens:
      prevState = prevState or {}
      local hs = {}
      local cs = {}
      -- go over time:
      for t = 1,steps do
         -- xt
         local xt = torch.select(x,2,t)

         -- prev h and prev c
         local hp = hs[t-1] or prevState.h or torch.zero(x.new(batch, hiddenFeatures))
         local cp = cs[t-1] or prevState.c or torch.zero(x.new(batch, hiddenFeatures))

         -- batch norm for t, independent mean and std across time steps
         local lstm_bn, cell_bn
         if batchNormalization then
           if layers.lstm_bn[t] then
             lstm_bn = layers.lstm_bn[t]
             cell_bn = layers.cell_bn[t]
           else
             -- all time steps beyond maxBatchNormalizationLayers uses the last one.
             lstm_bn = layers.lstm_bn[#layers.lstm_bn]
             cell_bn = layers.cell_bn[#layers.cell_bn]
           end
         end

         -- pack all dot products:
         local dots = torch.cat({xt, hp}, 2) * p.W

         if batchNormalization then
           -- use translation parameter from batch norm as bias.
           dots = lstm_bn({p.lstm_bn_1, p.lstm_bn_2}, dots)
         else
           dots = dots + torch.expand(p.b, batch, 4 * hiddenFeatures)
         end

         -- view as 4 groups:
         dots = torch.view(dots, batch, 4, hiddenFeatures)

         -- batch compute gates:
         local sigmoids = util.sigmoid( torch.narrow(dots, 2,1,3) )
         local inputGate = torch.select(sigmoids, 2,1)
         local forgetGate = torch.select(sigmoids, 2,2)
         local outputGate = torch.select(sigmoids, 2,3)

         -- write inputs
         local tanhs = torch.tanh( torch.narrow(dots, 2,4,1) )
         local inputValue = torch.select(tanhs, 2,1)

         -- next c:
         cs[t] = torch.cmul(forgetGate, cp) + torch.cmul(inputGate, inputValue)

         if batchNormalization then
           cs[t] = cell_bn({p.cell_bn_1, p.cell_bn_2}, cs[t])
         end

         -- next h:
         hs[t] = torch.cmul(outputGate, torch.tanh(cs[t]))
      end


      -- save state
      local newState = {h=hs[#hs], c=cs[#cs]}

      -- output:
      if outputType == 'last' then
         -- return last hidden code:
         return hs[#hs], newState, layers
      else
         -- return all:
         for i in ipairs(hs) do
            hs[i] = torch.view(hs[i], batch,1,hiddenFeatures)
         end
         return x.cat(hs, 2), newState, layers
      end
   end

   -- layers
   return f, params, layers
end
