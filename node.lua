local _ = require 'moses'

local nodeApply, getOutgrad, newStartNode, node

-- Declare the ops we'd like to override directly
local op = {
   add = function(a,b) return a+b end,
   sub = function(a,b) return a-b end,
   mul = function(a,b) return a*b end,
   div = function(a,b) return a/b end,
   pow = function(a,b) return a^b end,
   unm = function(a) return -1*a end -- TODO: more efficient way across numbers and torch?
}

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
   local _getValue
   _getValue = function(v)
      if isNode(v) then
         return v.value
      else
         return v
      end
   end
   return _getValue(v)
end

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

node = {
   op = op,
   Node = Node,
   isNode = isNode,
   getValue = getValue,
   nodeApply = nodeApply,
   getOutgrad = getOutgrad,
   checkInput = checkInput,
   newStartNode = newStartNode,
}
return node