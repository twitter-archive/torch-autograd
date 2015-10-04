Autograd
========

Autograd automatically differentiates native
[Torch](https://github.com/torch/torch7) code.

Scope
-----

Autograd has multiple goals:

* provide automatic differentiation of [Torch](https://github.com/torch/torch7)
  expressions
* support arbitrary Torch types (e.g. transparent and full support
  for CUDA-backed computations)
* full integration with [nn](https://github.com/torch/nn) modules: mix and match
  auto-differentiation with user-provided gradients
* represent complex evaluation graphs, which is very useful to describe models
  with multiple loss functions and/or inputs
* enable gradients of gradients for transparent computation of Hessians, ...

TODO
----

Autograd is work in progress. Current list of things to be developed includes:

* gradients of gradients (Hessian)
* auto-type for nn-wrapped primitives (type should be inferred at runtime, as
    is the case for autograd-generated code)
* add more useful examples of models
* implement auto-buffering so that native torch functions can re-use memory
  (i.e. auto-generate code that's similar to what nn does for modules)

Examples
--------

### Autograd example

A simple neural network with a multinomial logistic loss:

```lua
-- libraries:
t = require 'torch'
grad = require 'autograd'

-- define trainable parameters:
params = {
   W = {
      t.randn(100,50),
      t.randn(50,10),
   },
   b = {
      t.randn(50),
      t.randn(10),
   }
}

-- define model
neuralNet = function(params, x, y)
   local h1 = torch.tanh(x * params.W[1] + params.b[1])
   local h2 = torch.tanh(h1 * params.W[2] + params.b[2])
   local yHat = h2 - torch.log(torch.sum(torch.exp(h2)))
   local loss = - torch.sum(torch.cmul(yHat, y))
   return loss
end

-- gradients:
dneuralNet = grad(neuralNet)

-- some data:
x = t.randn(1,100)
y = t.Tensor(1,10):zero() y[1][3] = 1

-- compute loss and gradients wrt all parameters in params:
loss = neuralNet(params, x, y)
dparams = dneuralNet(params, x, y)

-- in this case:
--> loss: is a scalar (Lua number)
--> dparams: is a table that mimics the structure of params; for
--  each Tensor in params, dparams provides the derivatives of the
--  loss wrt to that Tensor.
```

See more complete examples in `examples/`.

### Gradient checks

For ease of mind (and to write proper tests), a simple grad checker
is provided. See `test.lua` for complete examples. In short, it can be
used like this:

```lua
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
```

### Wrapping nn modules

The [nn](https://github.com/torch/nn) library provides with all sorts of very optimized
primitives, with gradient code written and optimized manually. Sometimes it's useful
to rely on these for maximum performance.

Here we rewrite the neural net example from above, but this time relying on a mix of
`nn` primitives and `autograd`-inferred gradients:

```lua
-- libraries:
t = require 'torch'
grad = require 'autograd'

-- define trainable parameters:
params = {
   W = {
      t.randn(50,100), -- note that parameters are transposed (nn convention for nn.Linear)
      t.randn(10,50),
   },
   b = {
      t.randn(50),
      t.randn(10),
   }
}

-- instantiate nn primitives:
-- Note: we do this outside of the eval function, so that memory
-- is only allocated once; moving these calls to within the body
-- of neuralNet would work too, but would be quite slower.
linear1 = grad.nn.Linear(100, 50)
acts1 = grad.nn.Tanh()
linear2 = grad.nn.Linear(50, 10)
acts2 = grad.nn.Tanh()

-- define model
neuralNet = function(params, x, y)
   local h1 = acts1(linear1(x, params.W[1], params.b[1]))
   local h2 = acts2(linear2(h1, params.W[2], params.b[2]))
   local yHat = h2 - torch.log(torch.sum(torch.exp(h2)))
   local loss = - torch.sum(torch.cmul(yHat, y))
   return loss
end

-- gradients:
dneuralNet = grad(neuralNet)

-- some data:
x = t.randn(1,100)
y = t.Tensor(1,10):zero() y[1][3] = 1

-- compute loss and gradients wrt all parameters in params:
loss = neuralNet(params, x, y)
dparams = dneuralNet(params, x, y)

-- in this case:
--> loss: is a scalar (Lua number)
--> dparams: is a table that mimics the structure of params; for
    each Tensor in params, dparams provides the derivatives of the
    loss wrt to that Tensor.
```

