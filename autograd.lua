-- TODO
-- Disallow overwriting anything
-- Tables

-- Deps
local torch = require('torch')
local haveCutorch,cutorch = pcall(require,'cutorch')
local class = require 'class'
local debug = require 'debug'
local _ = require 'moses'
require 'pl'
require 'trepl'

-- Declare the ops we'd like to override directly
local op = {
   add = function(a,b) return a+b end,
   sub = function(a,b) return a-b end,
   mul = function(a,b) return a*b end,
   div = function(a,b) return a/b end,
   pow = function(a,b) return a^b end,
   unm = function(a) return -1*a end -- TODO: more efficient way across numbers and torch?
}

-- Some local declarations ahead of time
local gradfuns = {}
local nodeApply, getOutgrad

-- Define the tensor types for which we'll allow automatic differentiation
local tensorTypes = {
   'FloatTensor',
   'DoubleTensor'
}
if haveCutorch then
   tensorTypes[#tensorTypes+1] = 'CudaTensor'
end

-- Make a node class, which will capture computation as they're used
local noop = function() end
-- Define the node
local Node = {
   value=0,
   fun=nil,
   args={},
   tape={},
   outgrad=0,
   name=""
}

-- Niceties
function Node:__tostring()
   if type(self.value) == "table" then
      return pretty.write(self.value)
   else
      return tostring(self.value)
   end
end

function Node:__tostring()
   return tostring(self.value)
end

function Node.__add(l,r)
   return nodeApply(op.add, l, r)
end

function Node.__sub(l,r)
   return nodeApply(op.sub, l, r)
end

function Node.__mul(l,r)
   return nodeApply(op.mul, l, r)
end

function Node.__div(l,r)
   return nodeApply(op.div, l, r)
end
function Node.__pow(l,r)
   return nodeApply(op.pow, l, r)
end
function Node.__unm(l)
   return nodeApply(op.unm, l)
end


function Node:new(value, fun, args, tape, name)
   local o = {}
   setmetatable(o, self)

   o.tape = tape or {}
   o.tape[#o.tape+1] = o
   o.value = value
   o.fun = fun
   o.outgrad = 0.0
   o.args = args or {}
   o.name = "" or name
   return o
end

local function isNode(n)
   return getmetatable(n) == Node
end

local function getValue(v)
   if isNode(v) then
      return v.value
   else
      return v
   end
end

-- -- Proxy the table lookups.
-- function Node.__index(tbl,key)
--    if Node[key] or key == "fun" then
--       return Node[key]
--    else
--       local o = rawget(tbl, "value")
--       return o[key]
--    end
-- end
-- function Node.__newindex(tbl, k, v)
--    if tbl[k] then
--       rawset(tbl, k, v)
--    else
--       local o = rawget(tbl, "value")
--       if type(o) ~= "table" then error("Node's value is not a table type. Cannot set members.") end
--       rawset(o, k, v)
--    end
-- end


-- Override operations for number types
local numberMetatable = {
   __add = function(a,b)
      if torch.type(a) == "number" and torch.isTensor(b) then
         return nodeApply(op.add, b, a)
      else
         return nodeApply(op.add, a, b)
      end
   end,
   __sub = function(a,b)
      if torch.type(a) == "number" and torch.isTensor(b) then
         return nodeApply(op.add, -b, a)
      else
         return nodeApply(op.add, a, -b) -- TODO subtraction
      end
   end,
   __mul = function(a,b)
      if torch.type(a) == "number" and torch.isTensor(b) then
         return nodeApply(op.mul, b, a)
      else
         return nodeApply(op.mul, a, b)
      end
   end,
   __div = function(a,b)
      if torch.type(a) == "number" and torch.isTensor(b) then
         -- THIS IS INSANE
         c = torch.ones(b:size())
         return node.apply(op.mul, torch.cdiv(c,b), a)
      else
         return node.apply(op.div, a, b)
      end
   end,
   __unm = function(a,b)
      error("UNDEFINED")
   end
}
debug.setmetatable(1.0, numberMetatable)

-- A wrapper for a function
-- Anytime we try to apply a function to some arguments,
-- we'd like to make sure that if we're passing nodes in,
-- that we unpack the value in those nodes, apply the function
-- to the underlying value, and then wrap the value in a node
nodeApply = function(fun, ...)
   local _nodeApply
   _nodeApply = function(fun, ...)
      local arg = {...}
      local parents = _.filter(arg, function (k,v) return isNode(v) end)
      if _.count(parents) > 0 then
         local vals = _.map(arg,function(k,v) return getValue(v) end)
         local value = _nodeApply(fun,unpack(_.map(arg, function(k,v) return getValue(v) end)))
         return Node:new(value, fun, arg, parents[1].tape)
      else
         return fun(unpack(_.map(arg,function (k,v) return getValue(v) end)))
      end
   end
   return _nodeApply(fun, ...)
end

-- If we passed in just a tensor, return the outgrad.
-- If we passed in a table, return all the outgrads.
getOutgrad = function(arg)
   local _getOutgrad
   _getOutgrad = function(arg)

      local val = getValue(arg)

      -- If we have a tensor, we just have one out gradient
      if torch.isTensor(val) then
         return arg.outgrad

         -- If we have a table, then we can recurse the table and spit out the gradient
      elseif type(val) == "table" and not isNode(val) then
         local out = {}
         for k,v in pairs(arg) do
            out[k] = _getOutgrad(v)
         end
         return out
      end
   end
   return _getOutgrad(arg)
end

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

-- local newStartNode
newStartNode = function(val, tape)
   -- If our argument is a tensor, just nodify it straight-up
   if torch.isTensor(val) then
      return Node:new(val, nil, nil, tape)
      -- If our target argument is a table, we'll need to walk its members and node-ify them.
   elseif type(val) == "table" then
      for k,v in pairs(val) do
         val[k] = newStartNode(v, tape)
      end
      return val
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

-- Package
local autograd = {
   VERSION = '0.1',
   LICENSE = 'MIT',
   grad = grad,
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
