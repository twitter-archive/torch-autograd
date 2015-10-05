-- TODO
-- Disallow overwriting anything
-- Tables

-- Deps
local torch = require('torch')
local haveCutorch,cutorch = pcall(require,'cutorch')
local debug = require 'debug'
local _ = require 'moses'
local node = require 'autograd.node'
local nodeApply = node.nodeApply
local getOutgrad = node.getOutgrad
local checkInput = node.checkInput
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
if haveCutorch then
   tensorTypes[#tensorTypes+1] = 'CudaTensor'
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
      local ans = fun(unpack(arg))
      if not isNode(ans) then
         return 0.0
      end
      if type(getValue(ans)) ~= "number" then
         print("")
         print("Autograd only supports scalar outputs. This is current functions output: ")
         print(getValue(ans))
         error("Autograd only supports scalar return values. Output is not scalar")
      end

      local fnNames = _.map(ans.tape,function(k,t)
         if t.fun then
            return gradfuns[t.fun][1]
         else
            return nil
         end
      end)

      ans.outgrad = 1.0

      local node
      for i=#ans.tape,1,-1 do
         node = ans.tape[i]
         for iarg=1,#node.args do
            local thisArg = node.args[iarg]
            if isNode(thisArg) then
               local gradfun = gradfuns[node.fun][iarg+1]
               local gradUpdate = gradfun(node.outgrad, unpack(_.map(node.args, function(k,v) return getValue(v) end)))
               thisArg.outgrad = thisArg.outgrad + gradUpdate
               if thisArg.fun then
                  thisArg.name = gradfuns[thisArg.fun][1]
               else
                  thisArg.name = "data"
               end
            end
         end
      end

      -- Now spit out the grads
      if returnTape then
         return getOutgrad(arg[argnum]), ans.value, ans.tape
      else
         return getOutgrad(arg[argnum]), ans.value
      end
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
   function(g, x, y) return g end,
   function(g, x, y) return g end,
}
gradfuns[op.mul] = {
   "mult/dot",
   function(g, A, B)
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
   function (g, A, B)
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
   function(g, x) return -g end
}
gradfuns[op.div] = {
   "div",
   function(g, x, y) return elemwiseDiv(g,y) end,
   function(g, x, y) return elemwiseMul(-g,elemwiseDiv(x,torch.pow(y,2))) end,
}
gradfuns[op.sub] = {
   "sub",
   function(g, x, y) return g end,
   function(g, x, y) return -g end,
}
gradfuns[op.pow] = {
   "pow",
   function(g, x, y) return elemwiseMul(elemwiseMul(g,y),torch.pow(x, (y-1))) end,
}
gradfuns[torch.add] = {
   "torch.add",
   function(g, x, y) return g end,
   function(g, x, y) return g end,
}
gradfuns[torch.cmul] = {
   "torch.cmul",
   function(g, x, y) return elemwiseMul(y,g) end,
   function(g, x, y) return elemwiseMul(x,g) end,
}
gradfuns[torch.mul] = {
   "torch.mul",
   function(g, x, y) return elemwiseMul(y,g) end,
   function(g, x, y) return elemwiseMul(x,g) end,
}
gradfuns[torch.pow] = {
   "torch.pow",
   function(g, x, y) return elemwiseMul(elemwiseMul(g,y),torch.pow(x,y-1)) end
}
gradfuns[torch.exp] = {
   "exp",
   function(g,x) return elemwiseMul(torch.exp(x), g) end,
}
gradfuns[torch.tanh] = {
   "tanh",
   function(g,x) return elemwiseDiv(g,torch.pow(torch.cosh(x), 2.0)) end
}
gradfuns[torch.abs] = {
   "abs",
   function(g,x)
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

gradfuns[torch.sum] = {
   "sum",
   function(g,x,axis)
      local repeater = repeatToMatchShape(x, axis)
      return repeater(g)
   end
}
gradfuns[torch.sqrt] = {
   "sqrt",
   function(g,x) return elemwiseMul(elemwiseMul(g,0.5), torch.pow(x,-0.5)) end
}
gradfuns[torch.sin] = {
   "sin",
   function(g,x) return elemwiseMul(g, torch.cos(x)) end
}
gradfuns[torch.cos] = {
   "cos",
   function(g,x) return elemwiseMul(g, -torch.sin(x)) end
}
gradfuns[torch.tan] = {
   "tan",
   function(g,x) return elemwiseDiv(g, torch.pow(torch.cos(x), 2.0)) end
}
gradfuns[torch.log] = {
   "log",
   function(g,x) return elemwiseDiv(g,x) end
}


local override = {
   "__add", "add",
   "__mul", "mul", "cmul",
   "__unm", "__sub",
   "pow", "__pow",
   "exp", 'tanh',
   "sin", "cos", "tan", "sqrt",
   "abs", "sum", "log", "viewAs"
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
