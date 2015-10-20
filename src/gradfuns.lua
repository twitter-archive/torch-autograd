local gradfuns = { }

-- Helps with resizing gradients
-- Could also be called sumToMatchShape
local function unbroadcast(g,ans,x)
   if torch.isTensor(x) then
      if x:isSameSizeAs(g) then
         return g
      end

      if g:nElement() == x:nElement() then
         return torch.viewAs(g,x)
      end

      local size = torch.totable(x:size())
      local ndim = x:nDimension()
      local grad = g

      while grad:nDimension() > ndim do
         grad = torch.view(torch.sum(grad,1), thisSize)
      end

      -- If we're now the same size, then awesome
      if grad:nElement() == x:nElement() then
         return torch.viewAs(grad,x)

      -- Otherwise, we might have to sum across
      -- dimensions that are singleton for x,
      -- but not yet for the gradient
      else
      for i=1,#size do
            thisSize = torch.totable(grad:size())
         if size[i] == 1 then
               thisSize[i] = 1
               grad = torch.view(torch.sum(grad,i),unpack(thisSize))
            end
         end
         return grad
      end
   elseif torch.isTensor(ans) then
      return torch.sum(g)
   else
      return g
   end
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

gradfuns["op.add"] = {
   "add",
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(g,ans,y) end,
}
gradfuns["op.mul"] = {
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
gradfuns["op.unm"] = {
   "negation",
   function(g, ans, x) return -g end
}
gradfuns["op.div"] = {
   "div",
   function(g, ans, x, y) return unbroadcast(elemwiseDiv(g,y),ans,x) end,
   function(g, ans, x, y) return unbroadcast(elemwiseMul(-g,elemwiseDiv(x,torch.pow(y,2))),ans,y) end,
}
gradfuns["op.sub"] = {
   "sub",
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(-g,ans,y) end,
}
gradfuns["op.pow"] = {
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
   function(g, ans, x)
      local xx = torch.cosh(x)
      xx:cmul(xx)
      return elemwiseDiv(g, xx)
   end
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

return gradfuns