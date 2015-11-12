-- util
local util = require 'autograd.util'

return function(opt, params)
   -- options:
   opt = opt or {}
   local inputFeatures = opt.inputFeatures or 10
   local hiddenFeatures = opt.hiddenFeatures or 100
   local outputType = opt.outputType or 'last' -- 'last' or 'all'

   -- container:
   params = params or {}

   -- parameters:
   local p = {
      W = torch.zeros(inputFeatures+hiddenFeatures, 4 * hiddenFeatures),
      b = torch.zeros(1, 4 * hiddenFeatures),
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
      local hs = {}
      local cs = {}
      -- go over time:
      for t = 1,steps do
         -- xt
         local xt = torch.select(x,2,t)

         -- prev h and prev c
         local hp = hs[t-1] or prevState.h or torch.zero(x.new(batch, hiddenFeatures))
         local cp = cs[t-1] or prevState.c or torch.zero(x.new(batch, hiddenFeatures))

         -- pack all dot products:
         local dots = torch.cat(xt,hp,2) * p.W + torch.expand(p.b, batch, 4*hiddenFeatures)

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

         -- next h:
         hs[t] = torch.cmul(outputGate, torch.tanh(cs[t]))
      end


      -- save state
      local newState = {h=hs[#hs], c=cs[#cs]}

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

