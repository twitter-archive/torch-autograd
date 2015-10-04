-- Base package
local autograd = require 'autograd.main'

-- Meta info
autograd.VERSION = '0.1'
autograd.LICENSE = 'MIT'

-- Sub packages:
autograd.nn = require 'autograd.nnfuncwrapper'

-- Return package
return autograd
