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

local function optimizing()
   return defaultOptimize
end

local defaultProtected = false
local function protected(prot)
   defaultProtected = prot
end

local function grad(fn, gradOpt)
   gradOpt = gradOpt or { }
   local opt = util.deepCopy(gradOpt)
   opt.argnum = opt.gradArg or 1
   opt.optimize = util.defaultBool(opt.optimize, defaultOptimize)
   opt.protected = util.defaultBool(opt.protected, defaultProtected)
   opt.reduceFootprint = util.defaultBool(opt.reduceFootprint, false)
   opt.withForward = util.defaultBool(opt.withForward, true)
   opt.withGradients = util.defaultBool(opt.withGradients, true)
   opt.partialGrad = util.defaultBool(opt.partialGrad, false)
   if opt.optimize then
      return RuntimeCodegen.create(fn, opt)
   else
      if opt.stableGradients then
         error("stable gradient tensors only available in optimized mode")
      end
      return RuntimeDirect.create(fn, opt)
   end
end

-- Main functions:
local autograd = {
   grad = grad,
   overload = overload,
   optimize = optimize,
   optimizing = optimizing,
   protected = protected
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
