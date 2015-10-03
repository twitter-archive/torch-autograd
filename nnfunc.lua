-- TODO
-- Test that params are untouched
-- Test that it can be run multiple times
-- Test that it can be

local autograd = require 'autograd'
local nn = require 'nn'

local nnfunc = {}
function nnfunc.Linear(x, W, b)
   local forward, backward
   local grads = {}

   local _W = autograd._node.getValue(W)
   local linearModule = nn.Linear(_W:size(1), _W:size(2)):float()

   function forward(x,W,b)
      linearModule.weight:copy(W)
      linearModule.bias:copy(b)
      return linearModule:updateOutput(x)
   end

   function backward(arg,g,x,W,b)
      if not grads[arg] then
         linearModule:zeroGradParameters()
         linearModule:updateGradInput(x,g)
         linearModule:accGradParameters(x,g)
         grads['x'] = linearModule.gradInput
         grads['W'] = linearModule.gradWeight
         grads['b'] = linearModule.gradBias
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

   return autograd._node.nodeApply(forward, x, W, b)
end

return nnfunc
