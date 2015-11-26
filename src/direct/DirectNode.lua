local isTensor = torch.isTensor
local getOutgrad, newStartNode, node

local DirectNode = { }

function DirectNode:init(value, fun, gradFun, args, values, tape)
   local o = {}
   setmetatable(o, self)
   o.tape = tape
   tape[tape.nextIndex] = o
   tape.nextIndex = tape.nextIndex + 1
   o.value = value
   o.fun = fun
   o.gradFun = gradFun
   o.args = args
   o.argValues = values
   o.size = function(self, ...)
      return self.value.size(self.value,...)
   end
   o.dim = function(self, ...)
      return self.value.dim(self.value,...)
   end
   o.new = function(...)
      return o.value.new(...)
   end
   return o
end

function DirectNode.isNode(n)
   return getmetatable(n) == DirectNode
end

function DirectNode.getValue(v)
   if (getmetatable(v) == DirectNode) then
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
function DirectNode.nodeApply(fun, gradFun, capture, ...)
   fun = fun.fn
   local arg = {...}
   local parent = nil
   local values = { }
   local ln = #arg
   for k = 1, ln do
      local v = arg[k]
      if getmetatable(v) == DirectNode then
         parent = v
         values[#values + 1] = v.value
      elseif type(v) == "table" then
         local tableValue = {}
         for j,element in pairs(v) do
            if getmetatable(element) == DirectNode then
               parent = element
               tableValue[j] = element.value
            else
               tableValue[j] = element
            end
         end
         values[#values + 1] = tableValue
      else
         values[#values + 1] = v
      end
   end
   if capture and parent ~= nil then
      local value = fun(table.unpack(values))
      local node = nil
      local tape = parent.tape
      local o = tape[tape.nextIndex]
      if o ~= nil then
         o.tape = tape
         o.value = value
         o.fun = fun
         o.gradFun = gradFun
         o.args = arg
         o.outgrad = nil
         o.argValues = values
         tape.nextIndex = tape.nextIndex + 1
         return o
      end
      return DirectNode:init(value, fun, gradFun, arg, values, tape)
   else
      return fun(table.unpack(values))
   end
end

-- If we passed in just a tensor, return the outgrad.
-- If we passed in a table, return all the outgrads.
function DirectNode.getOutgrad(arg)
   local val = DirectNode.getValue(arg)
   -- If we have a tensor, we just have one out gradient
   if isTensor(val) then
      return arg.outgrad
      -- If we have a table, then we can recurse the table and spit out the gradient
   elseif type(val) == "table" and not (getmetatable(val) == DirectNode) then
      local out = {}
      for k,v in pairs(arg) do
         out[k] = DirectNode.getOutgrad(v)
      end
      return out
   end
end

-- local newStartNode
function DirectNode.newStartNode(val, tape)
   -- If our argument is a tensor, just nodify it straight-up
   if isTensor(val) then
      return DirectNode:init(val, nil, nil, { }, { }, tape)
      -- If our target argument is a table, we'll need to walk its members and node-ify them.
   elseif type(val) == "table" then
      local valCopy = { }
      for k,v in pairs(val) do
         valCopy[k] = DirectNode.newStartNode(v, tape)
      end
      return valCopy
   end
end

function DirectNode:__index(i)
   local value = rawget(self, "value")
   if torch.isTensor(value) and value[i] ~= nil then
      return value[i]
   end
   return rawget(DirectNode, i)
end

-- These exist only to be overloaded and called with flattened tensor or number arguments

function DirectNode.__add(a, b)
   return a + b
end

function DirectNode.__sub(a, b)
   return a - b
end

function DirectNode.__mul(a, b)
   return a * b
end

function DirectNode.__div(a, b)
   return a / b
end

function DirectNode.__pow(a, b)
   return a ^ b
end

function DirectNode.__unm(a)
   return -a
end

return DirectNode
