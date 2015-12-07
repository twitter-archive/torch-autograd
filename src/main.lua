local overload = require 'autograd.overload'
local RuntimeDirect = require 'autograd.runtime.direct'
local RuntimeCodegen = require 'autograd.runtime.codegen'
local util = require 'autograd.util'

-- Support functions
include 'support.lua'

-- Standard overloaded functions with gradients
include 'gradfuns.lua'

local defaultOptimize = false
local function optimize(opt)
   defaultOptimize = opt
end

local function grad(fn, gradOpt)
   gradOpt = gradOpt or { }
   local argnum = gradOpt.gradArg or 1
   local optimize = util.defaultBool(gradOpt.optimize, defaultOptimize)
   local withForward = util.defaultBool(gradOpt.withForward, true)
   local withGradients = util.defaultBool(gradOpt.withGradients, true)
   local partialGrad = util.defaultBool(gradOpt.partialGrad, false)
   local debugHook = gradOpt.debugHook
   local signatureFn = gradOpt.signatureFn
   local opt = {
      argnum = argnum,
      withForward = withForward,
      withGradients = withGradients,
      partialGrad = partialGrad,
      debugHook = debugHook,
      signatureFn = signatureFn
   }
   if optimize then
      return RuntimeCodegen.create(fn, opt)
   else
      return RuntimeDirect.create(fn, opt)
   end
end

-- Main functions:
local autograd = {
   grad = grad,
   overload = overload,
   optimize = optimize
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
