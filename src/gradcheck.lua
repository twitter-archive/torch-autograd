-- Autograd
local autograd = require 'autograd'
local __ = require 'moses'

-- Perturbation (finite diffs):
local perturbation = 1e-6

-- Threshold:
local threshold = 1e-5

-- Find grad:
local function findGrad(ref, x, dst)
   ref = __.flatten(ref)
   dst = __.flatten(dst)
   for i,v in ipairs(ref) do
      if v == x then
         return dst[i]
      end
   end
end

-- Compute grads with bprop:
local function jacobianFromAutograd(func, input, var)
   -- Eval to get output size
   local output = func(input)

   -- Autograd:
   local grads = autograd(func)(input)

   -- Find grad:
   local g = findGrad(input, var, grads)

   -- Return grads:
   return g:contiguous():view(-1):clone()
end

-- Compute grads from finite differences
local function jacobianFromFiniteDifferences(func, input, var)
   -- Flat view:
   local view = var:view(-1)

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
   if var then
      -- Random input:
      var:uniform(-1,1)

      -- Estimate grads with fprop:
      local jacobian1 = jacobianFromFiniteDifferences(func, input, var)

      -- Coded grads:
      local jacobian2 = jacobianFromAutograd(func, input, var)

      -- Error:
      local err = (jacobian1 - jacobian2):abs():max()

      -- Threhold?
      local pass = err < threshold
      if not pass then
         print('error = ' .. err)
      end
      return pass
   else
      -- get all vars:
      local vars = __.flatten(input)
      local ok = true
      for i,var in ipairs(vars) do
         ok = ok and gradcheck(func, input, var)
      end
      return ok
   end
end

-- Return package
return gradcheck
