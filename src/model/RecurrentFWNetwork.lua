-- util
local util = require 'autograd.util'
local functionalize = require('autograd.nnwrapper').functionalize
local nn = functionalize('nn')

return function(opt, params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 10
   local hiddenFeatures = opt.hiddenFeatures or 100
   local l = opt.lambda or 0.9
   local e = opt.eta or 0.5
   local S = opt.S or 1
   local LayerNorm = opt.LayerNorm or true
   local eps = eps or 1e-5
   local outputType = opt.outputType or 'last' -- 'last' or 'all'
   local relu = nn.ReLU()
   local mm  = nn.MM(false, false) -- A * B
   local mmT = nn.MM(false,  true) -- A * B'

   -- container:
   params = params or {}

   -- parameters:
   local p = {
      W = torch.zeros(inputFeatures+hiddenFeatures, hiddenFeatures),
      b = torch.zeros(1, hiddenFeatures),
   }
   table.insert(params, p)

   -- function:
   local f = function(params, x, prevState)
      -- dims:
      local p = params[1] or params
      if torch.nDimension(x) == 2 then
         x = torch.view(x, 1, torch.size(x, 1), torch.size(x, 2))
      end
      local batch = torch.size(x, 1)
      local steps = torch.size(x, 2)

      -- hiddens:
      prevState = prevState or {}

      -- prev h
      local hp = prevState.h or torch.zero(x.new(batch, hiddenFeatures))

      -- fast weights
      local A = prevState.A or
                torch.zero(x.new(batch, hiddenFeatures, hiddenFeatures))

      local hs = {}
      -- go over time:
      for t = 1, steps do
         -- xt
         local xt = torch.select(x, 2, t)

         -- prev h
         hp = hs[t-1] or hp

         -- vector to matrix
         local hpMat = torch.view(hp, batch, -1, 1)

         -- fast weights update
         A = l * A + e * mmT{hpMat, hpMat}

         -- pack all dot products:
         local dot = torch.cat(xt, hp, 2) * p.W
                   + torch.expand(p.b, batch, hiddenFeatures)

         hs[t] = torch.zero(x.new(batch, hiddenFeatures))
         for s = 0, S do

            -- vector to matrix
            local hstMat = torch.view(hs[t], batch, -1, 1)

            -- next h:
            hs[t] = dot + torch.view(mm{A, hstMat}, batch, -1)

            if LayerNorm then
               local h = hs[t]
               if torch.nDimension(hs[t]) == 1 then
                  h = torch.view(hs[t], 1, torch.size(hs[t], 1))
               end
               local n = torch.size(h, 2)
               h = h - torch.expand(torch.mean(h, 2), torch.size(h))
               local std = torch.expand(
                  torch.sqrt(torch.sum(torch.cmul(h, h) / n, 2) + eps),
               torch.size(h))
               hs[t] = torch.view(torch.cdiv(h, std), torch.size(hs[t]))
            end

            -- apply non-linearity
            hs[t] = relu(hs[t])

         end
      end


      -- save state
      local newState = {h=hs[#hs]}

      -- output:
      if outputType == 'last' then
         -- return last hidden code:
         return hs[#hs], newState
      else
         -- return all:
         for i in ipairs(hs) do
            hs[i] = torch.view(hs[i], batch,1,hiddenFeatures)
         end
         return x.cat(hs, 2), newState
      end
   end

   -- layers
   return f, params
end

