local gradfuns = require 'autograd.gradfuns'
local node = require 'autograd.node'
local isTensor = require 'autograd.util'.isTensor
local Node = node.Node
local nodeApply = node.nodeApply
local getOutgrad = node.getOutgrad
local newStartNode = node.newStartNode
local isNode = node.isNode
local getValue = node.getValue

local tapeRecordingDepth = 0
local debugFns = { }

-- Define the tensor types for which we'll allow automatic differentiation
-- util.isTensor must also be updated
local tensorTypes = {
   'FloatTensor',
   'DoubleTensor'
}
if haveCutorch and cutorch then
   tensorTypes[#tensorTypes+1] = 'CudaTensor'
end

local override = {
   "__add", "add",
   "__mul", "mul", "cmul",
   "__unm", "__sub",
   "__div","div","cdiv",
   "inverse",
   "pow", "__pow",
   "mean", "var", "std",
   "exp", 'tanh',
   "sin", "cos", "tan", "sqrt",
   "abs", "sum", "log", "viewAs", "view", "expand", "expandAs",
   "select", "narrow", "min", "max", "cat", "index", "clone", -- "copy"
}

local op = {
   add = function(a,b) return a+b end,
   sub = function(a,b) return a-b end,
   mul = function(a,b) return a*b end,
   div = function(a,b) return a/b end,
   pow = function(a,b) return a^b end,
   unm = function(a) return -a end
}

local function nodeOperator(name)
   local fun = op[name]
   local gradFun = gradfuns["op."..name]
   return function(l, r)
      if tapeRecordingDepth == 0 then
         return fun(l, r)
      end
      return nodeApply(fun, gradFun, l, r)
   end
end

local function nodeUnaryOperator(name)
   local fun = op[name]
   local gradFun = gradfuns["op."..name]
   return function(l)
      if tapeRecordingDepth == 0 then
         return fun(l)
      end
      return nodeApply(fun, gradFun, l)
   end
end

Node.__add = nodeOperator("add")
Node.__sub = nodeOperator("sub")
Node.__mul = nodeOperator("mul")
Node.__div = nodeOperator("div")
Node.__pow = nodeOperator("pow")
Node.__unm = nodeUnaryOperator("unm")

-- Now override class methods and metamethods on tensors
-- Override metamethods like __mul and __add
local elemOpOverride = {
   __mul = "op.mul",
   __sub = "op.sub",
   __div = "op.div",
   __add = "op.add",
   __unm = "op.unm",
}

-- Override operations for number types
local numberMetatable = {
   __add = function(a,b)
      if type(a) == "number" and isTensor(b) then
         return nodeApply(op.add, gradfuns["op.add"], b, a)
      else
         return nodeApply(op.add, gradfuns["op.add"], a, b)
      end
   end,
   __sub = function(a,b)
      if type(a) == "number" and isTensor(b) then
         return nodeApply(op.sub, gradfuns["op.sub"], -b, a)
      else
         return nodeApply(op.sub, gradfuns["op.sub"], a, -b) -- TODO subtraction
      end
   end,
   __mul = function(a,b)
      if type(a) == "number" and isTensor(b) then
         return nodeApply(op.mul, gradfuns["op.mul"], b, a)
      else
         return nodeApply(op.mul, gradfuns["op.mul"], a, b)
      end
   end,
   __div = function(a,b)
      if type(a) == "number" and isTensor(b) then
         -- THIS IS INSANE
         c = torch.ones(b:size())
         return nodeApply(op.mul, gradfuns["op.mul"], torch.cdiv(c,b), a)
      else
         return nodeApply(op.div,  gradfuns["op.div"], a, b)
      end
   end,
   __unm = function(a,b)
      error("UNDEFINED")
   end
}

local shimTorch = { }
local origTorch = { }
local shimTorchClasses = { }
local origTorchClasses = { }

-- First, override all the Torch functions
for ifn,fnName in pairs(override) do
   local old = torch[fnName]
   local gradFn = gradfuns[old]
   if old ~= nil then
      local newFn = function(...)
         if tapeRecordingDepth == 0 then
            return old(...)
         end
         return nodeApply(old, gradFn, ...)
      end
      shimTorch[fnName] = newFn
      origTorch[fnName] = old
   end
end

for _,tensorType in pairs(tensorTypes) do
   local mt = torch.getmetatable('torch.' .. tensorType)
   for src,dest in pairs(elemOpOverride) do
      gradfuns[mt[src]] = gradfuns[dest]
   end
end

-- Make sure that all class functions
-- hook into the autodiff engine
-- (so that we capture evaluations of torch.sum() and also myTensor:sum())
for _,tensorType in pairs(tensorTypes) do
   local mt = torch.getmetatable('torch.' .. tensorType)
   shimTorchClasses[tensorType] = { }
   origTorchClasses[tensorType] = { }
   for ifn,fnName in pairs(override) do
      local old = mt[fnName]
      local gradFn = gradfuns[old]
      if old ~= nil then
         local newFn = function(...)
            if tapeRecordingDepth == 0 then
               return old(...)
            end
            return nodeApply(old, gradFn, ...)
         end
         shimTorchClasses[tensorType][fnName] = newFn
         origTorchClasses[tensorType][fnName]  = old
      end
   end
end

local function install()
   debug.setmetatable(1.0, numberMetatable)
   for fnName,ifn in pairs(shimTorch) do
      torch[fnName] = ifn
   end
   for _,tensorType in pairs(tensorTypes) do
      local mt = torch.getmetatable('torch.' .. tensorType)
      local fns = shimTorchClasses[tensorType]
      for fnName,ifn in pairs(fns) do
         rawset(mt, fnName, ifn)
      end
   end
end

local function uninstall()
   debug.setmetatable(1.0, nil)
   for fnName,ifn in pairs(origTorch) do
      torch[fnName] = ifn
   end
   for _,tensorType in pairs(tensorTypes) do
      local mt = torch.getmetatable('torch.' .. tensorType)
      local fns = origTorchClasses[tensorType]
      for fnName,ifn in pairs(fns) do
         rawset(mt, fnName, ifn)
      end
   end
end

local function defineGradient(obj, fnName, gradFn)
   local old = obj[fnName]
   obj[fnName] = function(...)
      if tapeRecordingDepth == 0 then
         return old(...)
      end
      return nodeApply(old, gradFn, ...)
   end
end

local function beginRecording()
   tapeRecordingDepth = tapeRecordingDepth + 1
end

local function endRecording()
   tapeRecordingDepth = tapeRecordingDepth - 1
end

-- Main functions:
local overload = {
   debugFns = debugFns,
   install = install,
   uninstall = uninstall,
   beginRecording = beginRecording,
   endRecording = endRecording,
   tensorTypes = tensorTypes,
   isTensor = isTensor,
   defineGradient = defineGradient
}

-- Return package
return overload

