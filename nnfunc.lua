-- TODO
-- Test that params are untouched
-- Test that it can be run multiple times
-- Test that it can be

local autograd = require 'autograd'
local nn = require 'nn'

local nnfunc = {}
function nnfunc.Linear(W, b, x)
   local forward, backward
   local grads = {}

   -- TODO: automatic casting
   local _W = autograd._node.getValue(W)
   local linearModule = nn.Linear(_W:size(1), _W:size(2)):float()

   function forward(W, b, x)
      linearModule.weight:copy(W)
      linearModule.bias:copy(b)
      return linearModule:updateOutput(x)
   end

   function backward(arg,g,W,b,x)
      if not grads[arg] then
         linearModule:updateGradInput(x, g)
         linearModule:accGradParameters(x, g)
         grads["W"] = linearModule.gradWeight
         grads["b"] = linearModule.gradBias
         grads["x"] = linearModule.gradInput
      end
      return grads[arg]
   end

   autograd.gradfuns[forward] = {
      "Linear",
      function(g,W,b,x)
         return backward("W",g,W,b,x)
      end,
      function(g,W,b,x)
         return backward("b",g,W,b,x)
      end,
      function(g,W,b,x)
         return backward("x",g,W,b,x)
      end
   }

   return autograd._node.nodeApply(forward, W, b, x)
end

return nnfunc
