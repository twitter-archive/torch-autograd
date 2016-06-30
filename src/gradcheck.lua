-- Autograd
local autograd = require 'autograd'

-- Perturbation (finite diffs):
local perturbation = 1e-6

-- Threshold:
local threshold = 1e-5

-- Compute grads with bprop:
local function jacobianFromAutograd(func, inputs, key)
   -- Autograd:
   local df = autograd(func)
   local grads = df(table.unpack(inputs))
   local gradsVerify = df(table.unpack(inputs))

   -- Find grad:
   local g = autograd.util.nestedGet(grads, key)
   local gVerify = autograd.util.nestedGet(gradsVerify, key)
   local err
   if torch.isTensor(g) then
      err = (g - gVerify):abs():max()
   else
      err = torch.abs(g - gVerify)
   end

   if err ~= 0 then
      error("autograd gradient not deterministic")
   end

   -- Return grads:
   if torch.isTensor(g) then
      return g:contiguous():view(-1):clone()
   else
      return g
   end
end

-- Compute grads from finite differences
local function jacobianFromFiniteDifferences(func, inputs, key)
   local var = autograd.util.nestedGet(inputs[1], key)

   if torch.isTensor(var) then
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
         local pred1 = func(table.unpack(inputs))
         view[i] = val + perturbation/2
         local pred2 = func(table.unpack(inputs))
         view[i] = val

         -- Finite diff:
         grads[i] = (pred2-pred1) / perturbation
      end
      -- Return grads:
      return grads
   else
      -- Initial val:
      local val = var

      -- Perturbate:
      autograd.util.nestedSet(inputs[1], key, val - perturbation/2)
      local pred1 = func(table.unpack(inputs))
      autograd.util.nestedSet(inputs[1], key, val + perturbation/2)
      local pred2 = func(table.unpack(inputs))
      autograd.util.nestedSet(inputs[1], key, val)

      -- Finite diff:
      return (pred2-pred1) / perturbation
   end
end

local function gradcheckvar2(func, inputs, key, randomizeInput)
   local var = autograd.util.nestedGet(inputs[1], key)
   local isTensorVar = torch.isTensor(var)

   -- Random input:
   if randomizeInput then
      if isTensorVar then
         var:uniform(-10,10)
      else
         autograd.util.nestedSet(inputs[1], key, 20 * (math.random() - 0.5))
         var = autograd.util.nestedGet(inputs[1], key)
      end
   end

   -- Estimate grads with fprop:
   local jacobian = jacobianFromAutograd(func, inputs, key)
   local originalLoss = func(table.unpack(inputs))
   
   local perturbedLoss, approxPerturbed

   if isTensorVar then
      local noise = jacobian:view(-1):clone():zero()
      local idx = math.random(1, noise:size(1))
      local originalVar = var:clone()
      noise:narrow(1,idx,1):uniform(-perturbation, perturbation)
      var:add(torch.view(noise, var:size()))

      perturbedLoss = func(table.unpack(inputs))
      approxPerturbed = originalLoss + torch.dot(jacobian, noise)
      var:copy(originalVar)
   else
      local noise = 2*perturbation*(math.random()-0.5)
      autograd.util.nestedSet(inputs[1], key, var + noise)

      perturbedLoss = func(table.unpack(inputs))
      approxPerturbed = originalLoss + jacobian * noise
      autograd.util.nestedSet(inputs[1], key, var)
   end

   -- Error:
   local err = math.abs((perturbedLoss - approxPerturbed)) /
      (math.max(math.abs(perturbedLoss), math.abs(originalLoss))+perturbation)

   -- Threhold?
   local pass = err < threshold
   if not pass then
      print('original loss = '..originalLoss)
      print('perturbed loss = '..perturbedLoss)
      print('approximated perturbed loss = '..approxPerturbed)
      print('error = ' .. err)
   end
   return pass, err
end

local function gradcheckvar(func, inputs, key, randomizeInput)
   local var = autograd.util.nestedGet(inputs[1], key)
   local isTensorVar = torch.isTensor(var)

   -- Random input:
   if randomizeInput then
      if isTensorVar then
         var:uniform(-1,1)
      else
         autograd.util.nestedSet(inputs[1], key, 2*math.random()-1)
      end
   end

   -- Estimate grads with fprop:
   local jacobian1 = jacobianFromFiniteDifferences(func, inputs, key)

   -- Coded grads:
   local jacobian2 = jacobianFromAutograd(func, inputs, key)

   -- Error:
   local err
   if isTensorVar then
      err = (jacobian1 - jacobian2):abs():max()
   else
      err = torch.abs(jacobian1 - jacobian2)
   end

   -- Threhold?
   local pass = err < threshold
   if not pass then
      print('error = ' .. err)
   end
   return pass
end

-- Test grads:
return function(opt)
   -- Options
   local randomizeInput = opt.randomizeInput
   if randomizeInput == nil then
      randomizeInput = true
   end

   -- Run grad check:
   local function gradcheck(func, ...)
      local args = {...}
      -- get all vars:
      local vars, keys = autograd.util.sortedFlattenKeys(args[1])
      local max_err = 0
      local ok = true
      for i,key in ipairs(keys) do
         local t, err = gradcheckvar2(func, args, key, randomizeInput)
         ok = ok and t
         if err > max_err then max_err = err end
         ok = ok and gradcheckvar(func, args, key, randomizeInput)
      end
      print('[gradcheck2] maximum error = '..max_err)
      return ok
   end

   -- Grad check fun:
   return gradcheck
end
