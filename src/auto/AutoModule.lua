local auto = require 'autograd.auto'
local autograd = require 'autograd.main'

-- This generates a new autograd.nn.[moduleName]
-- that takes a suitable forward function executed in :updateOutput
-- it automatically deals with the updateGradInput and accGradParameters differentiation
return function(moduleName)
   moduleName = moduleName or error('You must provide a name for your auto-differentiated module.')
   local AutoModule, parent = torch.class('autograd.nn.'..moduleName, 'nn.Module', auto)
   local module = auto[moduleName]
   -- Constructor needs a function and params (if any)
   -- The input function has the format (for Linear + ReLU):
   -- function(input, weight, bias)
   --    local y = params.weight * input + params.bias
   --    local output = torch.mul( torch.abs( y ) + y, 0.5)
   --    return output
   -- end
   function module:__init(fn, weight, bias)
      parent.__init(self)
      self.fn = fn or error('An autograd function must be specified as input to AutoModule')
      self.weight,self.gradWeight = weight and weight, weight:clone()
      self.bias,self.gradBias = bias and bias, bias:clone()
      local fnWrapper = function(params)
         return self.fn(params.input, params.weight, params.bias)
      end
      self.f = autograd(fnWrapper, { withForward = true, withGradients = false })
      self.b = autograd(fnWrapper, { withForward = false, withGradients = true, partialGrad = true })
   end

   function forward(self, input)
      self.output = self.f(input)
      return self.output
   end

   function module:updateOutput(input)
      self.grads = nil
      self.output = self.f({input=input, weight=self.weight, bias=self.bias})
      return self.output
   end

   function module:updateGradInput(input, gradOutput)
      self.grads = self.b({input=input, weight=self.weight, bias=self.bias}, gradOutput)
      self.gradInput = self.grads.input
      return self.gradInput
   end

   function module:accGradParameters(input, gradOutput, scale)
      if not self.grads then
         self.grads = self.b({input=input, weight=self.weight, bias=self.bias})
      end
      self.gradWeight:add(scale, self.grads.weight)
      self.gradBias:add(scale, self.grads.bias)
   end

   return module
end
