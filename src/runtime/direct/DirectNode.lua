local isTensor = torch.isTensor
local getOutgrad, newStartNode, node

local DirectNode = { }

function DirectNode:init(value, fun, gradFun, args, values, tape)
   local o = {}
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
   o.view = function(...)
      return torch.view(...)
   end
   o.viewAs = function(...)
      return torch.viewAs(...)
   end
   o.expand = function(...)
      return torch.expand(...)
   end
   o.expandAs = function(...)
      return torch.expandAs(...)
   end
   setmetatable(o, self)
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
   elseif type(val) == "number" then
      return arg.outgrad
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
   elseif type(val) == "number" then
      return DirectNode:init(val, nil, nil, {}, {}, tape)
   end
end

function DirectNode.__internal_set(s, k, v)
   s[k] = v
   return s
end

function DirectNode.__internal_get(s, k)
   return s[k]
end

function DirectNode:__index(i)
   local value = rawget(self, "value")
   if torch.isTensor(value) and value[i] ~= nil then
      if type(i) ~= "string" then
         return DirectNode.__internal_get(self, i)
      end
   end
   return rawget(DirectNode, i)
end

function DirectNode:__newindex(k, v)
   local value = rawget(self, "value")
   if torch.isTensor(value) then
      if type(k) ~= "string" then
         return DirectNode.__internal_set(self, k, v)
      end
   end
   return rawset(self, k, v)
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
