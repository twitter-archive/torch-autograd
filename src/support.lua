-- Various torch additions, should move to torch itself

torch.select = function (A, dim, index)
   return A:select(dim, index)
end

torch.index = function (A, dim, index)
   return A:index(dim, index)
end

torch.narrow = function(A, dim, index, size)
   return A:narrow(dim, index, size)
end

torch.clone = function(A)
   local B = A.new(A:size())
   return B:copy(A)
end

torch.contiguous = function(A)
   return A:contiguous()
end

torch.copy = function(A,B)
   local o = A:copy(B)
   return o
end

torch.size = function(A, dim)
   return A:size(dim)
end

torch.nDimension = function(A)
   return A:nDimension()
end

torch.nElement = function(A)
   return A:nElement()
end

torch.isSameSizeAs = function(A, B)
   return A:isSameSizeAs(B)
end

torch.transpose = function(A)
   return A:t()
end

torch.long = function(A)
   return A:long()
end

torch.narrow = function(A, dim, index, size)
   return A:narrow(dim, index, size)
end

torch.typeAs = function(A, B)
   return A:type(B:type())
end

-- Allow number * tensor style operations

function unwrapNumberValue(v)
   if type(v) == 'table' then
      return v.raw
   else
      return v
   end
end

local numberMetatable = {
   __add = function(a,b)
      if type(a) == "number" and torch.isTensor(b) then
         return b + a
      elseif type(a) == "number" and type(b) == "table" then
         return a + unwrapNumberValue(b)
      else
         return a + b
      end
   end,
   __sub = function(a,b)
      if type(a) == "number" and torch.isTensor(b) then
         return -b + a
      elseif type(a) == "number" and type(b) == "table" then
         return a - unwrapNumberValue(b)
      else
         return a - b
      end
   end,
   __mul = function(a,b)
      if type(a) == "number" and torch.isTensor(b) then
         return b * a
      elseif type(a) == "number" and type(b) == "table" then
         return a * unwrapNumberValue(b)
      else
         return a * b
      end
   end
}

debug.setmetatable(1.0, numberMetatable)