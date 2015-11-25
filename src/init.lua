-- Deps
require 'trepl'

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
autograd.model = require 'autograd.model'
autograd.loss = require 'autograd.loss'
autograd.util = require 'autograd.util'

-- Return package
return autograd
