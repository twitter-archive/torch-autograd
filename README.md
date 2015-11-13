Autograd
========

Autograd automatically differentiates native
[Torch](https://github.com/torch/torch7) code. Inspired by the
[original Python version](https://www.github.com/hips/autograd).

Scope
-----

Autograd has multiple goals:

* provide automatic differentiation of [Torch](https://github.com/torch/torch7)
  expressions
* support arbitrary Torch types (e.g. transparent and full support
  for CUDA-backed computations)
* full integration with [nn](https://github.com/torch/nn) modules: mix and match
  auto-differentiation with user-provided gradients
* the ability to define any new nn compliant Module with automatic differentiation
* represent complex evaluation graphs, which is very useful to describe models
  with multiple loss functions and/or inputs
* enable gradients of gradients for transparent computation of Hessians, ...

Install
-------

* Install [Torch](http://www.torch.ch) (instructions [here](http://torch.ch/docs/getting-started.html#_)).
* Retrieve this repo
* Run: `luarocks make`


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
   local h1 = t.tanh(x * params.W[1] + params.b[1])
   local h2 = t.tanh(h1 * params.W[2] + params.b[2])
   local yHat = h2 - t.log(t.sum(t.exp(h2)))
   local loss = - t.sum(t.cmul(yHat, y))
   return loss
end

-- gradients:
dneuralNet = grad(neuralNet)

-- some data:
x = t.randn(1,100)
y = t.Tensor(1,10):zero() y[1][3] = 1

-- compute loss and gradients wrt all parameters in params:
dparams, loss = dneuralNet(params, x, y)

-- in this case:
--> loss: is a scalar (Lua number)
--> dparams: is a table that mimics the structure of params; for
--  each Tensor in params, dparams provides the derivatives of the
--  loss wrt to that Tensor.
```

Important note: only variables packed in the first argument of the
eval function will have their gradients computed. In the example above,
if the gradients wrt x are needed, then x simply has to be moved into
params. The params table can be arbitrarily nested.

See more complete examples in [examples](examples/).

Assuming the model defined above, and a training set of `{x,y}` pairs,
the model can easily be optimized using SGD:

```lua
for i,sample in datasetIterator() do
   -- estimate gradients wrt params:
   local grads, loss = dneuralNet(params, sample.x, sample.y)

   -- SGD step:
   for i = 1,#params.W do
      -- update params with an arbitrary learning rate:
      params.W[i]:add(-.01, grads.W[i])
      params.b[i]:add(-.01, grads.b[i])
   end
end
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
   local yHat = h2 - t.log(t.sum(t.exp(h2)))
   local loss = - t.sum(t.cmul(yHat, y))
   return loss
end

-- gradients:
dneuralNet = grad(neuralNet)

-- some data:
x = t.randn(1,100)
y = t.Tensor(1,10):zero() y[1][3] = 1

-- compute loss and gradients wrt all parameters in params:
dparams, loss = dneuralNet(params, x, y)
```

This code is stricly equivalent to the code above, but will be more efficient
(this is especially true for more complex primitives like convolutions, ...).

3rd party libraries that provide a similar API to nn can be
registered like this:

```lua
local customnnfuncs = grad.functionalize('customnn')  -- requires 'customnn' and wraps it
module = customnnfuncs.MyNnxModule(...)

-- under the hood, this is already done for nn:
grad.nn = grad.functionalize('nn')
```

On top of this functional API, existing `nn` modules and containers, with arbitarily
nested parameters, can also be wrapped into functions. This is particularly handy
when doing transfer learning from existing models:

```lua
-- Define a standard nn model:
local model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3, 16, 3, 3, 1, 1, 1, 1))
model:add(nn.Tanh())
model:add(nn.Reshape(16*8*8))
model:add(nn.Linear(16*8*8, 10))
model:add(nn.Tanh())
-- Note that this model could have been pre-trained, and reloaded from disk.

-- Functionalize the model:
local modelf, params = autograd.functionalize(model)

-- The model can now be used as part of a regular autograd function:
local loss = autograd.nn.MSECriterion()
neuralNet = function(params, x, y)
   local h = modelf(params, x)
   return loss(h, y)
end

-- Note: the parameters are always handled as an array, passed as the first
-- argument to the model function (modelf). This API is similar to the other
-- model primitives we provide (see below in "Model Primitives").
```

### Creating auto-differentiated nn modules

For those who have a training pipeline that heavily relies on the torch/nn API,
torch-autograd defines the `autograd.nn.AutoModule` and `autograd.nn.AutoCriterion` and  function. When given a `name`, it will create
a new class locally under autograd.auto.name. This class can be instantiated by providing a function, a weight, and a bias.
Here we show an example of writing a 2-layer fully-connected module and an MSE criterion using `AutoModule` and `AutoCriterion`:

Here we rewrite the neural net example from above, but this time relying on a mix of
`nn` primitives and `autograd`-inferred gradients:

```lua
-- Define functions for modules
-- Linear
local linear  = function(input, weight, bias)
   local y = weight * input + bias
   return y
end

-- Linear + ReLU
local linearReLU  = function(input, weight, bias)
   local y = weight * input + bias
   local output = torch.mul( torch.abs( y ) + y, 0.5)
   return output
end

-- Define function for criterion
-- MSE
local mse = function(input, target)
   local buffer = input-target
   return torch.sum( torch.cmul(buffer, buffer) ) / (input:dim() == 2 and input:size(1)*input:size(2) or input:size(1))
end

-- Input size, nb of hiddens
local inputSize, outputSize = 100, 1000

-- Define auto-modules and auto-criteria
-- and instantiate them immediately
local autoModel = nn.Sequential()
local autoLinear1ReLU = autograd.nn.AutoModule('AutoLinearReLU')(linearReLU, linear1.weight:clone(), linear1.bias:clone())
local autoLinear2 = autograd.nn.AutoModule('AutoLinear')(linear, linear2.weight:clone(), linear2.bias:clone())
autoModel:add( autoLinear1ReLU )
autoModel:add( autoLinear2 )
local autoMseCriterion = autograd.nn.AutoCriterion('AutoMSE')(mse)
-- At this point, print(autograd.auto) should yield
-- {
--   AutoLinearReLU : {...}
--   AutoMSE : {...}
--   AutoLinear : {...}
-- }

-- Define number of iterations and learning rate
local n = 100000
local lr = 0.001
local autoParams,autoGradParams = autoModel:parameters()
local unifomMultiplier = torch.Tensor(inputSize):uniform()

-- Train: this should learn how to approximate e^(\alpha * x)
-- with an mlp aith both auto-modules and regular nn
for i=1,n do
   autoModel:zeroGradParameters()
   local input = torch.Tensor(inputSize):uniform(-5,5):cmul(uniformMultiplier)
   local target = input:clone():exp()
   -- Forward
   local output = autoModel:forward(input)
   local mseOut = autoMseCriterion:forward(output, target)
   -- Backward
   local gradOutput = autoMseCriterion:backward(output, target)
   local gradInput = autoModel:backward(input, gradOutput)
   autoModel:accGradParameters(input, gradOutput)
   for i=1,#autoParams do
      autoParams[i]:add(-lr, autoGradParams[i])
   end
end
```

### Gradient checks

For ease of mind (and to write proper tests), a simple grad checker
is provided. See [test.lua](test.lua) for complete examples. In short, it can be
used like this:

```lua
-- Parameters:
local W = t.Tensor(32,100):normal()
local x = t.Tensor(100):normal()

-- Function:
local func = function(inputs)
   return t.sum(inputs.W * inputs.x)
end

-- Check grads:
tester:assert(gradcheck(func, {W=W, x=x}, W), 'incorrect gradients on W')
tester:assert(gradcheck(func, {W=W, x=x}, x), 'incorrect gradients on x')

-- If grads wrt all inpts must be checked, omit the last variable:
tester:assert(gradcheck(func, {W=W, x=x}), 'incorrect gradients on W and x')
```

### Model Primitives

To ease the construction of new models, we provide primitives to generate
standard models.

Each constructor returns 2 things:

* `f`: the function, can be passed to `grad(f)` to get gradients
* `params`: the list of trainable parameters

Once instantiated, `f` and `params` can be used like this:

```lua
input = torch.randn(10)
pred = f(params, input)
grads = autograd(f)(params, input)
```

Current list of model primitives includes:

#### autograd.model.NeuralNetwork

API:

```lua
f,params = autograd.model.NeuralNetwork({
   -- number of input features:
   inputFeatures = 10,

   -- number of hidden features, per layer, in this case
   -- 2 layers, each with 100 and 10 features respectively:
   hiddenFeatures = {100,10},

   -- activation functions:
   activations = 'ReLU',

   -- if true, then no activation is used on the last layer;
   -- this is useful to feed a loss function (logistic, ...)
   classifier = false,

   -- dropouts:
   dropoutProbs = {.5, .5},
})
```

#### autograd.model.SpatialNetwork

API:

```lua
f,params = autograd.model.SpatialNetwork({
   -- number of input features (maps):
   inputFeatures = 3,

   -- number of hidden features, per layer:
   hiddenFeatures = {16, 32},

   -- poolings, for each layer:
   poolings = {2, 2},

   -- activation functions:
   activations = 'Sigmoid',

   -- kernel size:
   kernelSize = 3,

   -- dropouts:
   dropoutProbs = {.1, .1},
})
```

#### autograd.model.RecurrentNetwork

API:

```lua
f,params = autograd.model.RecurrentNetwork({
   -- number of input features (maps):
   inputFeatures = 100,

   -- number of output features:
   hiddenFeatures = 200,

   -- output is either the last h at step t,
   -- or the concatenation of all h states at all steps
   outputType = 'last', -- or 'all'
})
```

#### autograd.model.RecurrentLSTMNetwork

API:

```lua
f,params = autograd.model.RecurrentLSTMNetwork({
   -- number of input features (maps):
   inputFeatures = 100,

   -- number of output features:
   hiddenFeatures = 200,

   -- output is either the last h at step t,
   -- or the concatenation of all h states at all steps
   outputType = 'last', -- or 'all'
})
```

### Loss Primitives

Similarly to model primitives, we provide common loss functions in
`autograd.loss`:

```lua
-- cross entropy between 2 vectors:
-- (for categorical problems, the target should be encoded as one-hot)
loss = loss.crossEntropy(prediction, target)

-- binary cross entropy - same as above, but labels are considered independent bernoulli variables:
loss = loss.binaryEntropy(prediction, target)

-- least squares - mean square error between 2 vectors:
loss = loss.leastSquares(prediction, target)
```

### Debugging and fine-grain control

Debugging hooks can be inserted when wrapping the function with `autograd`:

```lua
grad(f, {
   debugHook = function(debugger, msg)
      -- dump a dot representation of the graph:
      debugger.generateDot('result.dot')
   end
})
```

Finer-grain control over execution can also be achieved using these flags:

```lua
-- All of these options default to true:
grad(f, {
   withForward = true | false,    -- compute the forward path
   withGradients = true | false,  -- compute the gradients (after forward)
   partialGrad = true | false     -- partial grad means that d(f) expects grads wrt output
})

-- Running this:
pred = grad(f, {withForward=true, withGradients=false})(inputs)
-- is equivalent to:
pred = f(inputs)
-- ... but the function is compiled, and benefits from tensor re-use!
```

TODO
----

Autograd is work in progress. Current list of things to be developed includes:

- [ ] Gradients of gradients (Hessian)
- [ ] Add support for sparse gradients
- [x] Add support for caching tape for a given input configuration
- [x] Code generation
- [x] Implement auto-buffering so that native torch functions can re-use memory

License
-------

Licensed under the Apache License, Version 2.0.
[See LICENSE file](https://github.com/twitter/torch-autograd/blob/master/LICENSE).
