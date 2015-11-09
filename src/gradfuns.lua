local Value = require 'autograd.Value'
local util = require 'autograd.util'
local overload = require 'autograd.overload'

local function newTensor(t, s)
   if Value.isValue(t) then
      local cc = t:get().new(s)
      return cc
   else
      return t.new(s)
   end
end

-- Helps with resizing gradients
-- Could also be called sumToMatchShape
local function unbroadcast(g,ans,x)
   if torch.isTensor(x) then
      if torch.isSameSizeAs(x, g) then
         return g
      end

      if torch.nElement(g) == torch.nElement(x) then
         return torch.viewAs(g,x)
      end

      local size = torch.totable(x:size())
      local ndim = torch.nDimension(x)
      local grad = g

      while gratorch.nDimension(d) > ndim do
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
      return function(g) return util.fillSameSizeAs(x, _sum(g)) end, torch.nElement(x)
   else
      error("todo")
      size = x:size():fill(1)
      size[axis] = x:size(axis)
      return function(g) return torch.repeatTensor(g, size) end, size[axis]
   end
end

-- Shared functions

local functions = { }

functions.cat = {
   function(g, ans, x,y,dim)
      if torch.isTensor(x) then
         dim = dim or torch.nDimension(x)
         return torch.narrow(g, dim, 1, torch.size(x, dim))
      else
         -- Second argument is dimension if table is passed in
         error("todo")
         dim = y or x[1]:nDimension()
         local ln=#x
         local out = {}
         local currentIndex = 1
         for i=1,ln do
            local thisSize = x[i]:size(dim)
            out[i] = torch.narrow(g,dim,currentIndex,thisSize)
            currentIndex = currentIndex + thisSize
         end
         return out
      end
   end,
   function(g,ans,x,y,dim)
      dim = dim or x:nDimension()
      return torch.narrow(g, dim, x:size(dim)+1, y:size(dim))
   end
}

-- Shared operators

local operators = { }

operators.add = {
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(g,ans,y) end,
}
operators.mul = {
   function(g, ans, A, B)
      if torch.isTensor(A) and torch.isTensor(B) then
         if torch.nDimension(B) == 2 then
            return g*torch.transpose(B)
         elseif torch.nDimension(A) == 2 then
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
         if torch.nDimension(A) == 2 then
            return torch.transpose(A)*g
         elseif torch.nDimension(B) == 2 then
            return torch.ger(A, g)
         else
            return g*A
         end
      else
         return g*A
      end
   end,
}
operators.unm = {
   function(g, ans, x) return -g end
}
operators.div = {
   function(g, ans, x, y) return unbroadcast(elemwiseDiv(g,y),ans,x) end,
   function(g, ans, x, y) return unbroadcast(elemwiseMul(-g,elemwiseDiv(x,torch.pow(y,2))),ans,y) end,
}
operators.sub = {
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(-g,ans,y) end,
}
operators.pow = {
   function(g, ans, x, y)
      local newg = elemwiseMul(elemwiseMul(g,y),torch.pow(x,y-1))
      return unbroadcast(newg, ans, x)
   end,
   function(g, ans, x, y)
      local newg = elemwiseMul(g,elemwiseMul(torch.log(x),torch.pow(x,y)))
      return unbroadcast(newg, ans, y)
   end
}

overload.module("torch", torch, function(module)
   local tensorTypes = {"FloatTensor", "DoubleTensor"}
   for i = 1, #tensorTypes do
      local tt = tensorTypes[i]
      module.class(tt, function(class)
         for k, v in pairs(operators) do
            class.operator(k, v)
         end
         class.dynamic("new")
      end)
   end
   module.gradient("add", {
      function(g, ans, x, y) return unbroadcast(g,ans,x) end,
      function(g, ans, x, y) return unbroadcast(g,ans,y) end
   })
   module.gradient("cmul", {
      function(g, ans, x, y) return unbroadcast(elemwiseMul(y, g), ans, x) end,
      function(g, ans, x, y) return unbroadcast(elemwiseMul(x, g), ans, y) end,
   })
   module.gradient("mul", {
      function(g, ans, x, y) return unbroadcast(elemwiseMul(y,g),ans,x) end,
      function(g, ans, x, y) return unbroadcast(elemwiseMul(x,g),ans,y) end,
   })
   module.gradient("div", {
      function(g, ans, x, y) return unbroadcast(elemwiseDiv(g, y),ans,x) end,
      function(g, ans, x, y) return unbroadcast(elemwiseMul(-g, elemwiseDiv(x, torch.pow(y, 2))), ans, y) end,
   })
   module.gradient("cdiv", {
      function(g, ans, x, y) return unbroadcast(elemwiseDiv(g, y), ans, x) end,
      function(g, ans, x, y) return unbroadcast(elemwiseMul(-g,elemwiseDiv(x, torch.pow(y, 2))), ans, y) end,
   })
   module.gradient("pow", {
      function(g, ans, x, y)
         local newg = elemwiseMul(elemwiseMul(g,y),torch.pow(x,y-1))
         return unbroadcast(newg, ans, x)
      end,
      function(g, ans, x, y)
         local newg = elemwiseMul(g,elemwiseMul(torch.log(x),torch.pow(x,y)))
         return unbroadcast(newg, ans, y)
      end
   })
   module.gradient("inverse", {
      function(g, ans, x) return -((torch.transpose(ans) * g) * torch.transpose(ans)) end,
   })
   module.gradient("exp", {
      function(g, ans, x) return elemwiseMul(ans, g) end,
   })
   module.gradient("tanh", {
      function(g, ans, x)
         local cx = torch.cosh(x)
         local cxx = torch.cmul(x, x)
         return elemwiseDiv(g, cxx)
      end
   })
   module.gradient("abs", {
      function(g, ans, x)
         if torch.isTensor(x) then
            return elemwiseMul(g,torch.sign(x))
         else
            error("todo")
            sign = x>0 and 1 or x<0 and -1 or 0
            return elemwiseMul(g, sign)
         end
      end
   })
   module.gradient("contiguous", {
      function(g,ans,x)
         return g
      end
   })
   module.gradient("cat", functions.cat)
   module.gradient("expand", {
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
   })
   module.gradient("expandAs", {
      function(g, ans, x,template)
         local sizes = x:size():totable()
         local out = g
         for dim,size in pairs(sizes) do
            if size == 1 then
               out = torch.sum(out, dim)
            end
         end
         return out
      end,
      function(g, ans, x,template)
         return nil
      end
   })
   module.gradient("view", {
      function(g, ans, x,sizes)
         if not g:isContiguous() then
            g = g:contiguous()
         end
         return torch.view(g,x:size())
      end
   })
   module.gradient("viewAs", {
      function(g, ans, x,template)
         return torch.copy(torch.viewAs(g,x))
      end,
      function(g, ans, x,template)
         return nil -- g.new(template:size()):zero()
      end
   })
   module.gradient("clone", {
      function(g, ans, x)
         return g
      end,
   })
   module.gradient("copy", {
      function(g, ans, x, y)
         return g
      end,
      function(g, ans, x, y)
         return g
      end,
   })
   module.gradient("select", {
      function(g, ans, x,dim,index)
         error("todo, make functional, or move to a util method")
         local out = newTensor(g, x:size()):zero()
         local slice = out:select(dim,index)
         slice:copy(g)
         return out
      end
   })
   module.gradient("index", {
      function(g, ans, x,dim,index)
         error("todo, make functional, or move to a util method")
         local out = util.zerosLike(g, x)
         for i=1,index:size(1) do
            torch.narrow(out,dim,index[i],1):add(torch.narrow(g,dim,i,1))
         end
         return out
      end
   })
   module.gradient("narrow", {
      function(g, ans, x,dim,index,size)
         local out = util.zerosLike(g, x)
         local slice = util.narrowCopy(out, dim, index, size)
         return slice
      end
   })
   module.gradient("sum", {
      function(g, ans, x,axis)
         local repeater = repeatToMatchShape(x, axis)
         return repeater(g)
      end
   })
   module.gradient("mean", {
      function(g,ans,x,axis)
         local repeater,nrepeats = repeatToMatchShape(x,axis)
         return repeater(g)/nrepeats
      end
   })
   module.gradient("norm", {
      function(g,ans,x,p,dim)
         error("NOT IMPLEMENTED")
      end,
   })
   module.gradient("var", {
      function(g,ans,x,axis)
         error("NOT IMPLEMENTED")
         local repeater,nrepeats = repeatToMatchShape(x,axis)
      end
   })
   module.gradient("std", {
      function(g,ans,x,axis)
         error("NOT IMPLEMENTED")
         local repeater,nrepeats = repeatToMatchShape(x,axis)
      end
   })
   module.gradient("sqrt", {
      function(g, ans, x) return elemwiseMul(elemwiseMul(g,0.5), torch.pow(x,-0.5)) end
   })
   module.gradient("sin", {
      function(g, ans, x) return elemwiseMul(g, torch.cos(x)) end
   })
   module.gradient("cos", {
      function(g, ans, x) return elemwiseMul(g, -torch.sin(x)) end
   })
   module.gradient("tan", {
      function(g, ans, x) return elemwiseDiv(g, torch.pow(torch.cos(x), 2.0)) end
   })
   module.gradient("log", {
      function(g, ans, x) return elemwiseDiv(g,x) end
   })
   module.gradient("min", {
      function(g, ans, x,axis)
         local repeater = repeatToMatchShape(x,axis)
         local out = util.setNotEqual(x, repeater(ans), 0, repeater(g))
         return out
      end
   })
   module.gradient("max", {
      function(g, ans, x,axis)
         local repeater = repeatToMatchShape(x,axis)
         local out = util.setNotEqual(x, repeater(ans), 0, repeater(g))
         return out
      end
   })
   module.dynamic("ne",  "ger", "new", "fill", "zeros", "transpose", "cosh", "sign")
   module.static("size", "isTensor", "nDimension", "nElement", "isSameSizeAs")
end)

overload.module("Value", Value, function(module)
   for k, v in pairs(operators) do
      module.operator(k, v)
   end
end)

overload.module("util", util, function(module)
   module.dynamic("setNotEqual", "fillSameSizeAs", "zerosLike", "narrowCopy")
end)


