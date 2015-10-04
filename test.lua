-- Tester:
local _ = require 'moses'
local totem = require 'totem'
local autograd = require 'autograd'
local Node = autograd._node.Node
local isNode = autograd._node.isNode
local getValue = autograd._node.getValue
local gradcheck = require 'autograd.gradcheck'
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

   NodeClass = function()
      -- Build nodes
      local n = Node:new(3, nil, nil, {})
      local m = Node:new(torch.ones(10), nil, nil, {})

      -- Test we can identify nodes
      tester:asserteq(isNode(n), true, "Did not detect class properly")
      tester:asserteq(isNode(m), true, "Did not detect class properly")
      tester:asserteq(isNode({42}), false, "Did not detect class properly")
      tester:asserteq(isNode(3), false, "Did not detect class properly")
      tester:asserteq(isNode("hey"), false, "Did not detect class properly")
      tester:asserteq(isNode(torch.randn(10)), false, "Did not detect class properly")

      -- TODO: more thorough testing of tables that contain nodes
   end,

   NNFunc_Basic = function()
      -- Tested params:
      local inputSize = 100
      local outputSize = 50
      local W = torch.Tensor(outputSize,inputSize):normal()
      local b = torch.Tensor(outputSize):normal()
      local x = torch.Tensor(inputSize):normal()
      local params = {W=W, b=b, x=x}

      -- nn modules:
      local linear1 = autograd.nn.Linear(inputSize, outputSize)

      -- nn version:
      function f_nn(params)
         local funcout = linear1(params.x, params.W, params.b)
         return torch.sum(funcout)
      end

      -- autograd version:
      function f_autograd(params)
         return torch.sum(params.W * params.x + params.b)
      end

      -- Get the NN predictions
      local pred_nn = f_nn(params)
      local g_nn = autograd(f_nn)
      local grad_nn = g_nn(params)

      -- Get the autograd predictions
      local pred_autograd = f_autograd(params)
      local g_autograd = autograd(f_autograd)
      local grad_autograd = g_autograd(params)

      -- Check
      tester:asserteq((grad_nn.W-grad_autograd.W):abs():max(), 0, "Incorrect gradients")
      tester:asserteq((grad_nn.x-grad_autograd.x):abs():max(), 0, "Incorrect gradients")

      -- Run a 2nd time - gradients should get recomputed:
      W:normal()
      b:normal()
      x:normal()

      -- Get the NN predictions
      local pred_nn = f_nn(params)
      local g_nn = autograd(f_nn)
      local grad_nn = g_nn(params)

      -- Get the autograd predictions
      local pred_autograd = f_autograd(params)
      local g_autograd = autograd(f_autograd)
      local grad_autograd = g_autograd(params)

      -- Check
      tester:asserteq((grad_nn.W-grad_autograd.W):abs():max(), 0, "Incorrect gradients")
      tester:asserteq((grad_nn.x-grad_autograd.x):abs():max(), 0, "Incorrect gradients")
   end,

   NNFunc_MLP = function()
      -- More complex model:
      local inputSize = 100
      local hiddenSize = 50
      local outputSize = 10

      -- Trainable parameters:
      local x = torch.Tensor(inputSize):normal()
      local W1 = torch.Tensor(hiddenSize,inputSize):normal()
      local b1 = torch.Tensor(hiddenSize):normal()
      local W2 = torch.Tensor(outputSize,hiddenSize):normal()
      local b2 = torch.Tensor(outputSize):normal()
      local params = {W1=W1, b1=b1, W2=W2, b2=b2, x=x}

      -- nn modules:
      local linear1 = autograd.nn.Linear(inputSize, hiddenSize)
      local acts1 = autograd.nn.Tanh()
      local linear2 = autograd.nn.Linear(hiddenSize, outputSize)
      local acts2 = autograd.nn.Tanh()

      -- nn version:
      function mlp(params)
         local h1 = acts1(linear1(params.x, params.W1, params.b1))
         local h2 = acts2(linear2(h1, params.W2, params.b2))
         local o = torch.sum(h2)
         return o
      end

      -- Eval:
      local pred = mlp(params)
      local grads = autograd(mlp)(params)

      -- Check grads:
      tester:assert(gradcheck(mlp, params, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(mlp, params, 'W1'), 'incorrect gradients on W1')
      tester:assert(gradcheck(mlp, params, 'b1'), 'incorrect gradients on b1')
      tester:assert(gradcheck(mlp, params, 'W2'), 'incorrect gradients on W2')
      tester:assert(gradcheck(mlp, params, 'b2'), 'incorrect gradients on b2')
   end,

   NNFunc_CNN = function()
      -- Trainable parameters:
      local x = torch.Tensor(3, 8, 8):normal()
      local W1 = torch.Tensor(16, 3*5*5):normal()
      local b1 = torch.Tensor(16):normal()
      local W2 = torch.Tensor(100, 16*8*8):normal()
      local b2 = torch.Tensor(100):normal()
      local params = {W1=W1, b1=b1, W2=W2, b2=b2, x=x}

      -- nn modules:
      local conv1 = autograd.nn.SpatialConvolutionMM(3, 16, 5, 5, 1, 1, 2, 2)
      local acts1 = autograd.nn.Tanh()
      local flatten = autograd.nn.Reshape(16*8*8)
      local linear2 = autograd.nn.Linear(16*8*8, 100)
      local acts2 = autograd.nn.Tanh()

      -- nn version:
      function mlp(params)
         local h1 = acts1(conv1(params.x, params.W1, params.b1))
         local h2 = acts2(linear2(flatten(h1), params.W2, params.b2))
         local o = torch.sum(h2)
         return o
      end

      -- Eval:
      local pred = mlp(params)
      local grads = autograd(mlp)(params)

      -- Check grads:
      tester:assert(gradcheck(mlp, params, 'x'), 'incorrect gradients on x')
      tester:assert(gradcheck(mlp, params, 'W1'), 'incorrect gradients on W1')
      tester:assert(gradcheck(mlp, params, 'b1'), 'incorrect gradients on b1')
      tester:assert(gradcheck(mlp, params, 'W2'), 'incorrect gradients on W2')
      tester:assert(gradcheck(mlp, params, 'b2'), 'incorrect gradients on b2')
   end,
}

-- Run tests:
tester:add(tests):run()
