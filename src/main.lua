-- TODO
-- Disallow overwriting anything
-- Tables

-- Deps
local haveCutorch,cutorch = pcall(require,'cutorch')
local debug = require 'debug'
local node = require 'autograd.node'
local Node = node.Node
local nodeApply = node.nodeApply
local getOutgrad = node.getOutgrad
local newStartNode = node.newStartNode
local isNode = node.isNode
local getValue = node.getValue

require 'autograd.number'
require 'pl'
require 'trepl'

-- For debugging
local function printSize(a)
   if type(a) == "number" then
      print("1x1")
   elseif torch.isTensor(a) then
      print(torch.totable(a:size()))
   else
      print("???")
   end
end

-- Some local declarations ahead of time
local gradfuns = {}

local debugFns = {}
local tapeRecordingDepth = 0
local beginTapeRecording
local endTapeRecording

-- Define the tensor types for which we'll allow automatic differentiation
local tensorTypes = {
   'FloatTensor',
   'DoubleTensor'
}
if haveCutorch and cutorch then
   tensorTypes[#tensorTypes+1] = 'CudaTensor'
end

-- Make sure we've got the right thing going on
local function checkInput(arg)
   if torch.isTensor(arg) then
      local isValidType = false
      for _,tensorType in pairs(tensorTypes) do
         isValidType = isValidType or 'torch.' .. tensorType == torch.typename(arg)
      end
      if not isValidType then
         local errMsg = "Input tensor is invalid type " .. torch.typename(arg) .. ". Valid types are"
         for _, tensorType in pairs(tensorTypes) do
            errMsg = errMsg .. " " .. tensorType
         end
         error(errMsg)
      end
   end
end

-- Helps with resizing gradients
-- Could also be called sumToMatchShape
local function unbroadcast(g,ans,x)
   if torch.isTensor(x) then
      if x:isSameSizeAs(g) then
         return g
      end

      if g:nElement() == x:nElement() then
         return torch.viewAs(g,x)
      end

      local size = torch.totable(x:size())
      local ndim = x:nDimension()
      local grad = g

      while grad:nDimension() > ndim do
         grad = torch.view(torch.sum(grad,1), thisSize)
      end

      -- If we're now the same size, then awesome
      if grad:nElement() == x:nElement() then
         return torch.viewAs(grad,x)

      -- Otherwise, we might have to sum across
      -- dimensions that are singleton for x,
      -- but not yet for the gradient
      else
      for i=1,#size do
            thisSize = torch.totable(grad:size())
         if size[i] == 1 then
               thisSize[i] = 1
               grad = torch.view(torch.sum(grad,i),unpack(thisSize))
            end
         end
         return grad
      end
   elseif torch.isTensor(ans) then
      return torch.sum(g)
   else
      return g
   end
end

-- Step through the computation graph and find the gradient
local function grad(fun, argnum, returnTape)
   argnum = argnum or 1
   local doGrad = function(...)
      local arg = tablex.deepcopy({...})
      local tape = {}

      -- Check the argument, to make sure it's alright.
      checkInput(arg[argnum])

      -- If our target argument is a table, we'll need to walk its members and node-ify them.
      -- For now, if we see a number or a tensor, we'll node-ify it, otherwise,
      -- if it's a table, we'll try to walk it
      arg[argnum] = newStartNode(arg[argnum], tape)
      local allAns = {fun(unpack(arg))}
      local ans = allAns[1]
      if not isNode(ans) then
         error("A node type was not returned. This is either because a gradient was not defined, or the input is independent of the output")
      end
      if type(getValue(ans)) ~= "number" then
         print("")
         print("Autograd only supports scalar outputs. This is current functions output: ")
         print(getValue(ans))
         error("Autograd only supports scalar return values. Output is not scalar")
      end

      ans.outgrad = 1.0

      endTapeRecording()
      for i=#ans.tape,1,-1 do
         local node = ans.tape[i]
         if debugFns.preGradFn then
            debugFns.preGradFn(node)
         end
         for iarg=1,#node.args do
            local thisArg = node.args[iarg]
            if isNode(thisArg) then
               if node.outgrad == nil then
                  if torch.isTensor(node.value) then
                     node.outgrad = node.value.new(node.value:size()):zero()
                  elseif type(node.value) == "number" then
                     node.outgrad = 0.0
                  end
               end
               local gradUpdate = (node.gradFun[iarg+1])(node.outgrad, node.value, unpack(node.argValues))
               if thisArg.outgrad == nil or thisArg.outgrad == 0 then
                  thisArg.outgrad = gradUpdate
               else
                  thisArg.outgrad = thisArg.outgrad + gradUpdate
               end
            end
         end
         if debugFns.postGradFn then
            debugFns.postGradFn(node)
         end
      end
      beginTapeRecording()

      -- Now spit out the grads, along with any answers returned along the way
      local out = {}
      out[1] = getOutgrad(arg[argnum])

      local ansVal = getValue(allAns)
      if type(allAns) == "table" then
         for key,value in pairs(ansVal) do
            out[#out+1] = getValue(value)
         end
      else
         out[2] = ansVal
      end
      return unpack(out)
   end
   return doGrad
end

local function elemwiseMul(a,b)
   if torch.isTensor(a) and torch.isTensor(b) then
      return torch.cmul(a,b)
   else
      return a*b
   end
end

local function elemwiseDiv(a,b)
   if torch.isTensor(a) and torch.isTensor(b) then
      return torch.cdiv(a,b)
   else
      return a/b
   end
end

-- Declare the ops we'd like to override directly
local op = {
   add = function(a,b) return a+b end,
   sub = function(a,b) return a-b end,
   mul = function(a,b) return a*b end,
   div = function(a,b) return a/b end,
   pow = function(a,b) return a^b end,
   unm = function(a) return -a end
}

gradfuns[op.add] = {
   "add",
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(g,ans,y) end,
}
gradfuns[op.mul] = {
   "mult/dot",
   function(g, ans, A, B)
      if torch.isTensor(A) and torch.isTensor(B) then
         if B:nDimension() == 2 then
            return g*B:t()
         elseif A:nDimension() == 2 then
            return torch.ger(g, B) -- outer product
         else
            return g*B -- elemwiseMul required? what about 3D?
         end
      else
         return g*B
      end
   end,
   function(g, ans, A, B)
      if torch.isTensor(A) and torch.isTensor(B) then
         if A:nDimension() == 2 then
            return A:t()*g
         elseif B:nDimension() == 2 then
            return torch.ger(A, g)
         else
            return g*A
         end
      else
         return g*A
      end
   end,
}
gradfuns[op.unm] = {
   "negation",
   function(g, ans, x) return -g end
}
gradfuns[op.div] = {
   "div",
   function(g, ans, x, y) return unbroadcast(elemwiseDiv(g,y),ans,x) end,
   function(g, ans, x, y) return unbroadcast(elemwiseMul(-g,elemwiseDiv(x,torch.pow(y,2))),ans,y) end,
}
gradfuns[op.sub] = {
   "sub",
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(-g,ans,y) end,
}
gradfuns[op.pow] = {
   "pow",
   function(g, ans, x, y)
      local newg = elemwiseMul(elemwiseMul(g,y),torch.pow(x,y-1))
      return unbroadcast(newg, ans, x)
   end,
   function(g, ans, x, y)
      local newg = elemwiseMul(g,elemwiseMul(torch.log(x),torch.pow(x,y)))
      return unbroadcast(newg, ans, y)
   end
}
gradfuns[torch.add] = {
   "add",
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(g,ans,y) end,
}
gradfuns[torch.cmul] = {
   "cmul",
   function(g, ans, x, y) return unbroadcast(elemwiseMul(y,g),ans,x) end,
   function(g, ans, x, y) return unbroadcast(elemwiseMul(x,g),ans,y) end,
}
gradfuns[torch.mul] = {
   "mul",
   function(g, ans, x, y) return unbroadcast(elemwiseMul(y,g),ans,x) end,
   function(g, ans, x, y) return unbroadcast(elemwiseMul(x,g),ans,y) end,
}
gradfuns[torch.div] = {
   "div",
   function(g, ans, x, y) return unbroadcast(elemwiseDiv(g,y),ans,x) end,
   function(g, ans, x, y) return unbroadcast(elemwiseMul(-g,elemwiseDiv(x,torch.pow(y,2))),ans,y) end,
}
gradfuns[torch.cdiv] = {
   "cdiv",
   function(g, ans, x, y) return unbroadcast(elemwiseDiv(g,y),ans,x) end,
   function(g, ans, x, y) return unbroadcast(elemwiseMul(-g,elemwiseDiv(x,torch.pow(y,2))),ans,y) end,
}

gradfuns[torch.pow] = {
   "pow",
   function(g, ans, x, y)
      local newg = elemwiseMul(elemwiseMul(g,y),torch.pow(x,y-1))
      return unbroadcast(newg, ans, x)
   end,
   function(g, ans, x, y)
      local newg = elemwiseMul(g,elemwiseMul(torch.log(x),torch.pow(x,y)))
      return unbroadcast(newg, ans, y)
   end
}

gradfuns[torch.exp] = {
   "exp",
   function(g, ans, x) return elemwiseMul(ans, g) end,
}
gradfuns[torch.tanh] = {
   "tanh",
   function(g, ans, x)
      local xx = torch.cosh(x)
      xx:cmul(xx)
      return elemwiseDiv(g, xx)
   end
}
gradfuns[torch.abs] = {
   "abs",
   function(g, ans, x)
      if torch.isTensor(x) then
         return elemwiseMul(g,torch.sign(x))
      else
         sign = x>0 and 1 or x<0 and -1 or 0
         return elemwiseMul(g,sign)
      end
   end
}

local function _sum(x)
   if torch.isTensor(x) then
      return torch.sum(x)
   else
      return x
   end
end


local function repeatToMatchShape(x,axis)
   -- Special sum function to deal with numbers or tensors

   if not torch.isTensor(x) then
      return function(x) return x end, 1
   end

   local size
   if not axis then
      size = x:size()
      return function(g) return x.new(size):fill(_sum(g)) end, x:nElement()
   else
      size = x:size():fill(1)
      size[axis] = x:size(axis)
      return function(g) return torch.repeatTensor(g, size) end, size[axis]
   end
end


torch["select"] = function (A, dim, index)
   return A:select(dim, index)
end

torch["narrow"] = function(A, dim, index, size)
   return A:narrow(dim, index, size)
end

gradfuns[torch.cat] = {
   "cat",
   function(g, ans, x,y,dim)
      dim = dim or x:nDimension()
      return torch.narrow(g, dim, 1, x:size(dim))
   end,
   function(g, ans, x,y,dim)
      dim = dim or x:nDimension()
      return torch.narrow(g, dim, x:size(dim)+1, y:size(dim))
   end
}
gradfuns[torch.expand] = {
   "expand",
   function(g, ans, x,...)
      local xSizes = x:size():totable()
      local out = g
      for dim,size in pairs(xSizes) do
         if size == 1 then
            out = torch.sum(out,dim)
         end
      end
      return out
   end
}
gradfuns[torch.expandAs] = {
   "expandAs",
   function(g, ans, x,template)
      local sizes = x:size():totable()
      local out = g
      for dim,size in pairs(sizes) do
         if size == 1 then
            out = torch.sum(out,dim)
         end
      end
      return out
   end,
   function(g, ans, x,template)
      local o = g.new(template:size()):zero()
      return o
   end
}
gradfuns[torch.view] = {
   "view",
   function(g, ans, x,sizes)
      -- TODO: copy required?
      return torch.view(g,x:size())
   end
}
gradfuns[torch.viewAs] = {
   "viewAs",
   function(g, ans, x,template)
      -- TODO: copy required?
      return torch.viewAs(g,x)
   end,
   function(g, ans, x,template)
      return g.new(template:size()):zero()
   end
}


gradfuns[torch.select] = {
   "select",
   function(g, ans, x,dim,index)
      local out = g.new(x:size()):zero()
      local slice = out:select(dim,index)
      slice:copy(g)
      return out
   end
}
gradfuns[torch.narrow] = {
   "narrow",
   function(g, ans, x,dim,index,size)
      -- TODO: copy necessary here?
      local out = g.new(x:size()):zero()
      local slice = out:narrow(dim,index,size)
      slice:copy(g)
      return out
   end
}
gradfuns[torch.sum] = {
   "sum",
   function(g, ans, x,axis)
      local repeater = repeatToMatchShape(x, axis)
      return repeater(g)
   end
}
gradfuns[torch.mean] = {
   "mean",
   function(g,ans,x,axis)
      local repeater,nrepeats = repeatToMatchShape(x,axis)
      return repeater(g)/nrepeats
   end
}

gradfuns[torch.var] = {
   "var",
   function(g,ans,x,axis)
      error("NOT IMPLEMENTED, requires complex conjugate")
      local repeater,nrepeats = repeatToMatchShape(x,axis)
   end
}
gradfuns[torch.std] = {
   "std",
   function(g,ans,x,axis)
      error("NOT IMPLEMENTED, requires complex conjugate")
      local repeater,nrepeats = repeatToMatchShape(x,axis)
   end
}
gradfuns[torch.sqrt] = {
   "sqrt",
   function(g, ans, x) return elemwiseMul(elemwiseMul(g,0.5), torch.pow(x,-0.5)) end
}
gradfuns[torch.sin] = {
   "sin",
   function(g, ans, x) return elemwiseMul(g, torch.cos(x)) end
}
gradfuns[torch.cos] = {
   "cos",
   function(g, ans, x) return elemwiseMul(g, -torch.sin(x)) end
}
gradfuns[torch.tan] = {
   "tan",
   function(g, ans, x) return elemwiseDiv(g, torch.pow(torch.cos(x), 2.0)) end
}
gradfuns[torch.log] = {
   "log",
   function(g, ans, x) return elemwiseDiv(g,x) end
}
gradfuns[torch.min] = {
   "min",
   function(g, ans, x,axis)
      repeater = repeatToMatchShape(x,axis)
      -- ATTN: THIS IS PROBABLY NOT SMART.
      repeater = repeatToMatchShape(x,axis)
      local out = repeater(g)
      local mask = torch.ne(x,repeater(ans))
      out[mask] = 0
      return out
   end
}
gradfuns[torch.max] = {
   "max",
   function(g, ans, x,axis)
      repeater = repeatToMatchShape(x,axis)
      local out = repeater(g)
      local mask = torch.ne(x,repeater(ans))
      out[mask] = 0
      return out
   end
}

local override = {
   "__add", "add",
   "__mul", "mul", "cmul",
   "__unm", "__sub",
   "__div","div","cdiv",
   "pow", "__pow",
   "mean", "var", "std",
   "exp", 'tanh',
   "sin", "cos", "tan", "sqrt",
   "abs", "sum", "log", "viewAs", "view", "expand", "expandAs",
   "select", "narrow", "min", "max", "cat",
}

local function nodeOperator(name)
   local gradFun = gradfuns[name]
   return function(l, r)
      if tapeRecordingDepth == 0 then
         return name(l, r)
      end
      local debugObj = nil
      if debugFns.preFwdFn ~= nil then
         debugObj = debugFns.preFwdFn(gradFun[1], debugObj)
      end
      local value = nodeApply(name, gradFun, l, r)
      if debugFns.postFwdFn ~= nil then
         debugFns.postFwdFn(gradFun[1], debugObj)
      end
      return value
   end
end

local function nodeUnaryOperator(name)
   local gradFun = gradfuns[name]
   return function(l)
      if tapeRecordingDepth == 0 then
         return old(l)
      end
      local debugObj = nil
      if debugFns.preFwdFn ~= nil then
         debugObj = debugFns.preFwdFn(gradFun[1], debugObj)
      end
      local value = nodeApply(name, gradFun, l)
      if debugFns.postFwdFn ~= nil then
         debugFns.postFwdFn(gradFun[1], debugObj)
      end
      return value
   end
end

Node.__add = nodeOperator(op.add)
Node.__sub = nodeOperator(op.sub)
Node.__mul = nodeOperator(op.mul)
Node.__div = nodeOperator(op.div)
Node.__pow = nodeOperator(op.pow)
Node.__unm = nodeUnaryOperator(op.unm)

-- Now override class methods and metamethods on tensors
-- Override metamethods like __mul and __add
local elemOpOverride = {
   __mul = op.mul,
   __sub = op.sub,
   __div = op.div,
   __add = op.add,
   __unm = op.unm,
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
         local debugObj = nil
         if debugFns.preFwdFn ~= nil then
            debugObj = debugFns.preFwdFn(fnName, debugObj)
         end
         local value = nodeApply(old, gradFn, ...)
         if debugFns.postFwdFn ~= nil then
            debugFns.postFwdFn(fnName, debugObj)
         end
         return value
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
            local debugObj = nil
            if debugFns.preFwdFn ~= nil then
               debugObj = debugFns.preFwdFn(fnName, debugObj)
            end
            local value = nodeApply(old, gradFn, ...)
            if debugFns.postFwdFn ~= nil then
               debugFns.postFwdFn(fnName, debugObj)
            end
            return value
         end
         shimTorchClasses[tensorType][fnName] = newFn
         origTorchClasses[tensorType][fnName]  = old
      end
   end
end

local function installShim()
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

local function uninstallShim()
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

beginTapeRecording = function()
   tapeRecordingDepth = tapeRecordingDepth + 1
end

endTapeRecording = function()
   tapeRecordingDepth = tapeRecordingDepth - 1
end

installShim()
beginTapeRecording()

-- Main functions:
local autograd = {
   grad = grad,
   gradfuns = gradfuns,
   debugFns = debugFns,
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
