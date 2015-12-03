-- Base package
local autograd = require 'autograd.main'

-- Meta info
autograd.VERSION = '0.1'
autograd.LICENSE = 'Apache 2.0'

-- Sub packages:
autograd.nnwrapper = require 'autograd.nnwrapper'
autograd.functionalize = autograd.nnwrapper.functionalize
autograd.nn = autograd.functionalize('nn')
autograd.nn.AutoModule = require 'autograd.auto.AutoModule'
autograd.nn.AutoCriterion = require 'autograd.auto.AutoCriterion'
autograd.auto = require 'autograd.auto'
autograd.auto.factory = autograd.auto.factory or torch.factory

torch.factory = function(className)
   if className:find('autograd%.nn%.module%.') and autograd.auto.factory then
      autograd.nn.AutoModule(className:gsub('autograd%.nn%.module%.',''))
   end
   if className:find('autograd%.nn%.criterion%.') and autograd.auto.factory then
      autograd.nn.AutoCriterion(className:gsub('autograd%.nn%.criterion%.',''))
   end
   return autograd.auto.factory(className)
end

autograd.auto = require 'autograd.auto'
autograd.model = require 'autograd.model'
autograd.loss = require 'autograd.loss'
autograd.util = require 'autograd.util'
autograd.optim = require 'autograd.optim'

-- Return package
return autograd
