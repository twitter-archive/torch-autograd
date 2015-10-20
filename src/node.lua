local nodeApply, getOutgrad, newStartNode, node

-- Make a node class, which will capture computation as they're used
local Node = { }

-- Niceties
function Node:__tostring()
   if type(self.value) == "table" then
      return pretty.write(self.value)
   else
      return tostring(self.value)
   end
end

function Node:new(value, fun, gradFun, args, values, tape)
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
nodeApply = function(fun, gradFun, ...)
   local parent = nil
   local values = { }
   local ln = #arg
   for k = 1, ln do
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
      return Node:new(value, fun, gradFun, arg, values, tape)
   else
      return value
   end
end

-- If we passed in just a tensor, return the outgrad.
-- If we passed in a table, return all the outgrads.
getOutgrad = function(arg)
   local val = getValue(arg)

   -- If we have a tensor, we just have one out gradient
   if torch.isTensor(val) then
      return arg.outgrad

      -- If we have a table, then we can recurse the table and spit out the gradient
   elseif type(val) == "table" and not isNode(val) then
      local out = {}
      for k,v in pairs(arg) do
         out[k] = getOutgrad(v)
      end
      return out
   end
end

-- local newStartNode
newStartNode = function(val, tape)
   -- If our argument is a tensor, just nodify it straight-up
   if torch.isTensor(val) then
      return Node:new(val, nil, nil, { }, { }, tape)
      -- If our target argument is a table, we'll need to walk its members and node-ify them.
   elseif type(val) == "table" then
      local valCopy = { }
      for k,v in pairs(val) do
         valCopy[k] = newStartNode(v, tape)
      end
      return valCopy
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
