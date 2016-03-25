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
      W = torch.zeros(inputFeatures+hiddenFeatures, 2 * hiddenFeatures),
      b = torch.zeros(1, 2 * hiddenFeatures),
      V = torch.zeros(inputFeatures+hiddenFeatures, hiddenFeatures),
      c = torch.zeros(1, hiddenFeatures),
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
      -- go over time:
      for t = 1,steps do
         -- xt
         local xt = torch.select(x,2,t)

         -- prev h
         local hp = hs[t-1] or prevState.h or torch.zero(x.new(batch, hiddenFeatures))

         -- pack all dot products:
         local dots  = torch.cat(xt,hp,2) * p.W + torch.expand(p.b, batch, 2*hiddenFeatures)

         -- view as 2 groups:
         dots = torch.view(dots, batch, 2, hiddenFeatures)

         -- batch compute gates:
         local sigmoids = util.sigmoid( dots )
         local inputGate = torch.select(sigmoids, 2,1)
         local forgetGate = torch.select(sigmoids, 2,2)

         -- write inputs
         local inputValue = torch.tanh( torch.cat(xt,torch.cmul(hp,inputGate),2) * p.V + torch.expand(p.c, batch, hiddenFeatures) )

         -- next h:
         hs[t] = torch.cmul(1-forgetGate, hp) + torch.cmul(forgetGate, inputValue)
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

