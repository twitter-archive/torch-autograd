autograd = require 'autograd'
nn = require 'nn'

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
         print("SHOULD ONLY RUN ONCE")
         linearModule:accGradParameters(x, g)
         grads["W"] = linearModule.gradWeight
         grads["b"] = linearModule.gradBias
         grads["x"] = torch.zeros(x:size())
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

inputSize = 100
outputSize = 50
W = torch.FloatTensor(inputSize,outputSize):fill(.5)
b = torch.FloatTensor(outputSize):fill(0)
x = torch.FloatTensor(1,inputSize):fill(.5)
params = {W=W,b=b,x=x}

function f_nn(params)
   funcout = nnfunc.Linear(params.W, params.b, params.x)
   return torch.sum(funcout)
end

function f_autograd(params)
   return torch.sum(params.x * params.W + params.b)
end

print(torch.sum(x))
pred_nn = f_nn(params)
g_nn = autograd(f_nn)
grad_nn = g_nn(params)
print(torch.sum(x))

pred_autograd = f_autograd(params)
g_autograd = autograd(f_autograd)
grad_autograd = g_autograd(params)
print(torch.sum(x))

-- Test that params are untouched
print(torch.sum(grad_autograd.W))
print(torch.sum(grad_nn.W))