-- Base package
local autograd = require 'autograd.main'

-- Meta info
autograd.VERSION = '0.1'
autograd.LICENSE = 'MIT'

-- Sub packages:
autograd.functionalize = require 'autograd.nnwrapper'
autograd.nn.AutoModule = require 'autograd.auto.AutoModule'
autograd.nn.AutoCriterion = require 'autograd.auto.AutoCriterion'
autograd.auto = require 'autograd.auto'
autograd.model = require 'autograd.model'
autograd.loss = require 'autograd.loss'
autograd.util = require 'autograd.util'

-- Return package
return autograd
