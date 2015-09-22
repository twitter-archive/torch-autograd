-- Autograd
local autograd = require 'autograd'

-- Perturbation (finite diffs):
local perturbation = 1e-6

-- Threshold:
local threshold = 1e-6

-- Compute grads with bprop:
local function jacobianFromAutograd(func, input, var)
   -- Eval to get output size
   local output = func(input)

   -- Autograd:
   local grads = autograd(func)(input)

   -- Return grads:
   return grads[var]:view(-1):clone()
end

-- Compute grads from finite differences
local function jacobianFromFiniteDifferences(func, input, var)
   -- Flat view:
   local view = input[var]:view(-1)

   -- Grads:
   local grads = view:clone():zero()

   -- Finite diffs:
   for i = 1,view:size(1) do
      -- Initial val:
      local val = view[i]

      -- Perturbate:
      view[i] = val - perturbation/2
      local pred1 = func(input)
      view[i] = val + perturbation/2
      local pred2 = func(input)
      view[i] = val

      -- Finite diff:
      grads[i] = (pred2-pred1) / perturbation
   end

   -- Return grads:
   return grads
end

-- Test grads:
local function gradcheck(func, input, var)
   -- Random input:
   input[var]:uniform(-1,1)

   -- Estimate grads with fprop:
   local jacobian1 = jacobianFromFiniteDifferences(func, input, var)

   -- Coded grads:
   local jacobian2 = jacobianFromAutograd(func, input, var)

   -- Error:
   local err = (jacobian1 - jacobian2):abs():max()

   -- Threhold?
   return err < threshold
end

-- Return package
return gradcheck
