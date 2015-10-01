autograd = require 'autograd'
nn = require 'nn'

local function uniform(min, max, h, w) return torch.mul(torch.FloatTensor():rand(h,w), (max-min)) + min end

nnfunc = {}
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
         linearModule:accGradParameters(x, g)
         grads["W"] = linearModule.gradWeight
         grads["b"] = linearModule.gradBias
      end
      return grads[arg]
   end

   autograd.gradfuns[forward] = {
      "Linear",
      function(g,W,b,x) 
         print("W backwards")
         return backward("W",g,W,b,x) 
      end,
      function(g,W,b,x) 
         print("b backwards")
         return backward("b",g,W,b,x) 
      end,
      function(g,W,b,x)
         print("x backwards")
         return x:zeros(x:size())
      end
   }

   return autograd._node.nodeApply(forward, W, b, x)
end

inputSize = 100
outputSize = 50
torch.manualSeed(0)
W = uniform(-1/math.sqrt(outputSize),1/math.sqrt(outputSize), inputSize, outputSize)
b = torch.FloatTensor(outputSize):fill(0)
x = torch.randn(1, inputSize):float()
out = x*W + b

function f(params)
   funcout = nnfunc.Linear(params.W, params.b, params.x)
   return torch.sum(funcout)
end

pred = f({W=W,b=b,x=x})
g = autograd(f)
grad = g({W=W,b=b,x=x})