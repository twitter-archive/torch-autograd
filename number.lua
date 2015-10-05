local node = require 'autograd.node'
local nodeApply = node.nodeApply
local op = node.op

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
         return nodeApply(op.mul, torch.cdiv(c,b), a)
      else
         return nodeApply(op.div, a, b)
      end
   end,
   __unm = function(a,b)
      error("UNDEFINED")
   end
}
debug.setmetatable(1.0, numberMetatable)