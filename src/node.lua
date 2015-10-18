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


function Node:new(value, fun, args, values, tape, name)
   local o = {}
   setmetatable(o, self)

   o.tape = tape or {}
   o.tape[#o.tape+1] = o
   o.value = value
   o.fun = fun
   if torch.isTensor(value) then
      o.outgrad = value.new(value:size()):zero()
   elseif type(value) == "number" then
      o.outgrad = 0.0
   else
      error("Invalid value. Only numbers and tensors supported")
   end
   o.args = args or {}
   o.argValues = values or {}
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

-- A wrapper for a function
-- Anytime we try to apply a function to some arguments,
-- we'd like to make sure that if we're passing nodes in,
-- that we unpack the value in those nodes, apply the function
-- to the underlying value, and then wrap the value in a node
nodeApply = function(fun, ...)
   local _nodeApply
   _nodeApply = function(fun, ...)
      local arg = {...}
      local parent = nil
      local values = { }
      for k = 1, #arg do
         local v = arg[k]
         if getmetatable(v) == Node then
            parent = v
            values[#values + 1] = v.value
         else
            values[#values + 1] = v
         end
      end
      local value = fun(unpack(values))
      if parent ~= nil then
         return Node:new(value, fun, arg, values, parent.tape)
      else
         return value
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

-- local newStartNode
newStartNode = function(val, tape)
   -- If our argument is a tensor, just nodify it straight-up
   if torch.isTensor(val) then
      return Node:new(val, nil, nil, nil, tape)
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
   newStartNode = newStartNode,
}
return node
