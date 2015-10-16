-- TODO
-- Disallow overwriting anything
-- Tables

-- Deps
local haveCutorch,cutorch = pcall(require,'cutorch')
local debug = require 'debug'
local _ = require 'moses'
local node = require 'autograd.node'
local nodeApply = node.nodeApply
local getOutgrad = node.getOutgrad
local newStartNode = node.newStartNode
local isNode = node.isNode
local getValue = node.getValue
local op = node.op
require 'autograd.number'
require 'pl'
require 'trepl'


-- Some local declarations ahead of time
local gradfuns = {}

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
      local size = torch.totable(x:size())
      local ndim = x:nDimension()
      local out = g -- check that we do a copy!
      -- NOTE: I DO NOT KNOW HOW TO DO THIS IN ONE SHOT
      -- Get the overall dimensionality the same
      while g:nDimension() > ndim do
         out = torch.sum(out,1)
      end
      -- Now trim the gradient to match the singleton 
      -- dimensions in the input
      for i=1,#size do
         if size[i] == 1 then
            out = torch.sum(out,i)
         end
      end
      out = torch.viewAs(out,x)
      return out
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

      collectgarbage('stop')
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

      local node
      for i=#ans.tape,1,-1 do
         node = ans.tape[i]
         for iarg=1,#node.args do
            local thisArg = node.args[iarg]
            if isNode(thisArg) then
               local gradfun = gradfuns[node.fun][iarg+1]
               local thisArgs = {}
               for inodearg=1,#node.args do
                  thisArgs[inodearg] = getValue(node.args[inodearg])
               end
               thisArg.outgrad = thisArg.outgrad + gradfun(node.outgrad, node.value, unpack(thisArgs))
            end
         end
      end

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
      collectgarbage('restart')
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

local gradMt = {}
gradMt.__index = function(table, key)
   print("")
   print(debug.getinfo(key))
   error("No adjoint found for function " .. tostring(key) .. ". Debug info above.")
end
setmetatable(gradfuns, gradMt)

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
   function(g, ans, x) return elemwiseDiv(g,torch.pow(torch.cosh(x), 2.0)) end
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
      -- TODO: sparse tensors
      -- TODO: copy necessary here?
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

-- First, override all the Torch functions
for ifn,fnName in pairs(override) do
   local old = torch[fnName]
   local newFn = function(...)
      -- print("Running new " .. fnName)
      return nodeApply(old, ...)
   end
   torch[fnName] = newFn
end

-- Now override class methods and metamethods on tensors
-- Override metamethods like __mul and __add
local elemOpOverride = {
   __mul = op.mul,
   __sub = op.sub,
   __div = op.div,
   __add = op.add,
   __unm = op.unm,
}

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
   for ifn,fnName in pairs(override) do
      local old = mt[fnName]
      local newFn = function(...)
         -- print("Running metamethod " .. fnName)
         return nodeApply(old, ...)
      end
      rawset(mt, fnName, newFn)
   end
end

-- Main functions:
local autograd = {
   grad = grad,
   gradfuns = gradfuns,
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
