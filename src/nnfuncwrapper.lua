-- Register all nnfunc grads into autograd
local autograd = require 'autograd.main'
local node = require 'autograd.node'
local nnfunc = require 'nnfunc'

local loaded = {}

-- Generic auto-wrapper of every function exposed
-- by nnfunc:
local function functionalize(pkg)
   -- return pre-loaded package:
   if loaded[pkg] then
      return loaded[pkg]
   end

   -- get package from nnfunc:
   nnfunc.functionalize(pkg)

   -- rebundle package:
   local p = {}
   for name,Class in pairs(nnfunc[pkg]) do
      p[name] = function(...)
         -- instantiate nnfunc module:
         local eval = Class(...)

         -- return autograd evaluator:
         return function(x, W, b)
            local forward, backward
            local grads = {}

            function forward(x,W,b)
               local res = eval({input=x, weight=W, bias=b})
               return res.output
            end

            function backward(arg,g,x,W,b)
               if not grads[arg] then
                  local res = eval({
                     input=x,
                     weight=W, bias=b,
                     gradOutput = g,
                  })
                  grads['x'] = res.gradInput
                  grads['W'] = res.gradWeight
                  grads['b'] = res.gradBias
               end
               return grads[arg]
            end

            autograd.gradfuns[forward] = {
               "Linear",
               function(g,x,W,b)
                  return backward('x',g,x,W,b)
               end,
               function(g,x,W,b)
                  return backward('W',g,x,W,b)
               end,
               function(g,x,W,b)
                  return backward('b',g,x,W,b)
               end
            }

            return node.nodeApply(forward, x, W, b)
         end
      end
   end
   loaded[pkg] = p
   return p
end

return functionalize
