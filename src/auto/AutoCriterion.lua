local auto = require 'autograd.auto'
local d = require 'autograd.main'

-- This generates a new autograd.nn.[moduleName]
-- that takes a suitable forward function executed in :updateOutput
-- it automatically deals with the updateGradInput and accGradParameters differentiation
return function(criterionName)
   -- Input fn has the following format (MSE error):
   -- function(input, target)
   --    local buffer = input-target
   --    return torch.cmul(buffer, buffer) / (input:dim() == 2 and input:size(1) or 1)
   -- end
   local AutoCriterion, parent = torch.class('autograd.nn.'..criterionName, 'nn.Criterion', auto)
   criterionName = criterionName or error('You must provide a name for your auto-differentiated criterion.')
   local criterion = auto[criterionName]
   function criterion:__init(fn)
      parent.__init(self)
      self.fn = fn or error('An autograd function must be specified as input to AutoCriterion')
      self.fnWrapper = function(params, target)
         return self.fn(params.input, target)
      end
   end

   function criterion:updateOutput(input,y)
      self.gradInput, self.output, self.predictions = d(self.fnWrapper)({input=input}, y)
      return self.output
   end

   function criterion:updateGradInput(input, y)
      self.gradInput = self.gradInput.input
      return self.gradInput
   end

   return criterion
end
