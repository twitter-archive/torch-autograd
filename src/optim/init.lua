local moses = require 'moses'
local tablex = require 'pl.tablex'

local function wrap(optimfn)
   return function(fn, state, params)
      local states = { }
      local flatParams = moses.flatten(params)
      for i = 1, #flatParams do
         states[i] = tablex.copy(state)
      end
      return function(...)
         local out = {fn(params, ...)}
         grads, loss = out[1], out[2]
         local flatGrads = moses.flatten(grads)
         for i = 1, #flatGrads do
            local grad = flatGrads[i]
            optimfn(function()
               return loss, grad
            end, flatParams[i], states[i])
         end
         return table.unpack(out)
      end, states
   end
end

local opt = {}

for k, v in pairs(require 'optim') do
   opt[k] = wrap(v)
end

return opt