-- Tester:
local _ = require 'moses'
local totem = require 'totem'
local autograd = require 'autograd'
local gradcheck = require './gradcheck'
local tester = totem.Tester()

-- List of tests:
local tests = {
   Dot = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   GradCheck_Dot = function()
      -- Parameters:
      local W = torch.Tensor(32,100):normal()
      local x = torch.Tensor(100):normal()

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Check grads:
      tester:assert(gradcheck(func, {W=W, x=x}, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(func, {W=W, x=x}, 'W'), 'incorrect gradients on W')
   end,

   Scale = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x * 3.0 + 1.0)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 2432, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   GradCheck_Scale = function()
      -- Parameters:
      local W = torch.Tensor(32,100):normal()
      local x = torch.Tensor(100):normal()

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x * 3.0 + 1.0)
      end

      -- Check grads:
      tester:assert(gradcheck(func, {W=W, x=x}, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(func, {W=W, x=x}, 'W'), 'incorrect gradients on W')
   end,

   Unary = function()
      -- Parameters:
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(- inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({x=x})
      local grads = dFunc({x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, -50, 'incorrect prediction')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   DotNonLinear = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.tanh(inputs.W * inputs.x))
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 32, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   GradCheck_DotNonLinear = function()
      -- Parameters:
      local W = torch.Tensor(32,100):normal()
      local x = torch.Tensor(100):normal()

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.tanh(inputs.W * inputs.x))
      end

      -- Check grads:
      tester:assert(gradcheck(func, {W=W, x=x}, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(func, {W=W, x=x}, 'W'), 'incorrect gradients on W')
   end,

   FloatType = function()
      -- Parameters:
      local W = torch.FloatTensor(32,100):fill(.5)
      local x = torch.FloatTensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   CudaType = function()
      -- Cuda only:
      if not cutorch then
         return
      end

      -- Parameters:
      local W = torch.CudaTensor(32,100):fill(.5)
      local x = torch.CudaTensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,

   NCalls = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(inputs.W * inputs.x)
      end

      -- Grads:
      local dFunc = autograd(func)

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})
      for i = 1,5 do
         pred = func({W=W, x=x})
         grads = dFunc({W=W, x=x})
      end

      -- Tests:
      tester:asserteq(type(pred), 'number', 'incorrect prediction')
      tester:asserteq(pred, 800, 'incorrect prediction')
      tester:asserteq(grads.W:dim(), 2, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(1), 32, 'incorrect dims for gradients')
      tester:asserteq(grads.W:size(2), 100, 'incorrect dims for gradients')
      tester:asserteq(grads.x:dim(), 1, 'incorrect dims for gradients')
      tester:asserteq(grads.x:size(1),100, 'incorrect dims for gradients')
   end,
}

-- Run tests:
tester:add(tests):run()
