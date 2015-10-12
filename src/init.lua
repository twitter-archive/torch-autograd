-- Base package
local autograd = require 'autograd.main'

-- Meta info
autograd.VERSION = '0.1'
autograd.LICENSE = 'MIT'

-- Sub packages:
autograd.nn = require 'autograd.nnfuncwrapper'
autograd.model = require 'autograd.model'
autograd.loss = require 'autograd.loss'
autograd.util = require 'autograd.util'

-- Return package
return autograd
