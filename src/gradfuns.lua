local Value = require 'autograd.runtime.codegen.Value'
local DirectNode = require 'autograd.runtime.direct.DirectNode'
local util = require 'autograd.util'
local overload = require 'autograd.overload'

function getValue(v)
   if Value.isValue(v) then
      return v:get()
   elseif DirectNode.isNode(v) then
      return DirectNode.getValue(v)
   else
      return v
   end
end

-- Utility for defining gradients that are zero
local function zeroGradient(nArgs)
   nArgs = nArgs or 2
   zeroGrads = {}
   for i=1,nArgs do
      zeroGrads[i] = function(...) return nil end
   end
   return zeroGrads
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

      local size = torch.totable(torch.size(x))
      local ndim = torch.nDimension(x)
      local grad = g

      while torch.nDimension(grad) > ndim do
         grad = torch.view(torch.sum(grad,1), thisSize)
      end

      -- If we're now the same size, then awesome
      if torch.nElement(grad) == torch.nElement(x) then
         return torch.viewAs(grad,x)

      -- Otherwise, we might have to sum across
      -- dimensions that are singleton for x,
      -- but not yet for the gradient
      else
      for i=1,#size do
            thisSize = torch.totable(torch.size(grad))
         if size[i] == 1 then
               thisSize[i] = 1
               grad = torch.view(torch.sum(grad,i),table.unpack(thisSize))
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
      axis = getValue(axis)
      local size = torch.size(x):fill(1)
      size[axis] = torch.size(x, axis)
      return function(g) return torch.repeatTensor(g, size) end, size[axis]
   end
end

-- Shared functions

local functions = { }

functions.catGradient = {
   function(g, ans, x, y,dim)
      if torch.isTensor(x) then
         dim = dim or torch.nDimension(x)
         return torch.narrow(g, dim, 1, torch.size(x, dim))
      else
         if torch.isTensor(x[1]) then
            -- Second argument is dimension if table is passed in
            return util.catTableGradient(g, x, y)
         else
            return util.catNumberGradient(g, x, y)
         end
      end
   end,
   function(g,ans,x,y,dim)
      if torch.isTensor(y) then
         dim = dim or torch.nDimension(x)
         return torch.narrow(g, dim, torch.size(x, dim) + 1, torch.size(y, dim))
      else
         return nil
      end
   end
}

functions.get = {
   function(g, ans, x, k)
      local out = util.zerosLike(x)
      out[k] = g
      return out
   end,
   function(g, ans, x, k) return nil end,
}

functions.set = {
   function(g, ans, x, k, v)
      g[k] = 0
      return g
   end,
   function(g, ans, x, k, v)
      return nil
   end,
   function(g, ans, x, k, v)
      return g[k]
   end,
}

-- Shared operators

local operators = { }

operators.add = {
   function(g, ans, x, y) return unbroadcast(g,ans,x) end,
   function(g, ans, x, y) return unbroadcast(g,ans,y) end,
}
operators.mul = {
   function(g, ans, A, B)
      local isTensorA = torch.isTensor(A)
      local isTensorB = torch.isTensor(B)

      if not isTensorA and isTensorB then
         return torch.sum(elemwiseMul(g, B))
      elseif isTensorB and torch.nDimension(B) == 2 then
         return g * torch.t(B)
      elseif isTensorA and torch.nDimension(A) == 2 then
         if not isTensorB then
            return elemwiseMul(g, B)
         else
            return torch.ger(g,B)
         end
      else
         return B * g
      end
   end,
   function(g, ans, A, B)
      local isTensorA = torch.isTensor(A)
      local isTensorB = torch.isTensor(B)

      if not isTensorB and isTensorA then
         return torch.sum(elemwiseMul(g, A))
      elseif isTensorA and torch.nDimension(A) == 2 then
         return torch.t(A) * g
      elseif isTensorB and torch.nDimension(B) == 2 then
         if not isTensorA then
            return elemwiseMul(g, A)
         else
            return torch.ger(A, g)
         end
      else
         return A * g
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

-- Define some gradients that will be shared by the class and the torch module
-- e.g. torch.view(x,3,3) and x:view(3,3)
local viewGradients = {
   function(g, ans, x,sizes)
      return torch.view(util.makeContiguous(g), torch.size(x))
   end
}
local viewAsGradients = {
   function(g, ans, x,template)
      return torch.clone(torch.viewAs(g,x))
   end,
   function(g, ans, x,template)
      return nil -- g.new(template:size()):zero()
   end
}
local expandGradients = {
   function(g, ans, x,...)
      local xSizes = torch.size(x):totable()
      local out = g
      for dim,size in pairs(xSizes) do
         if size == 1 then
            out = torch.sum(out,dim)
         end
      end
      return out
   end
}
local expandAsGradients = {
   function(g, ans, x, template)
      local sizes = torch.size(x):totable()
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
}


overload.module("torch", torch, function(module)
   local tensorTypes = {"FloatTensor", "DoubleTensor", "CudaTensor"}
   for i = 1, #tensorTypes do
      local tt = tensorTypes[i]
      if torch[tt] ~= nil then
         module.class(tt, function(class)
            for k, v in pairs(operators) do
               class.operator(k, v)
            end
            class.gradient("cat", functions.catGradient)
            class.initializer("new")
            class.static("dim", "size", "nDimension", "nElement")
            class.gradient("view", viewGradients)
            class.gradient("viewAs", viewAsGradients)
            class.gradient("expand", expandGradients)
            class.gradient("expandAs", expandAsGradients)
            class.defaultUnsupported()
         end)
      end
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
   module.gradient("ger", {
      -- Only takes 1D vectors as input
      function(g, ans, x, y) return g * y end,
      function(g, ans, x, y) return torch.t(g) * x end
   })
   module.gradient("inverse", {
      function(g, ans, x) return -((torch.t(ans) * g) * torch.t(ans)) end,
   })
   module.gradient("exp", {
      function(g, ans, x) return elemwiseMul(ans, g) end,
   })
   module.gradient("tanh", {
      function(g, ans, x)
         local mzz = 1 - torch.cmul(ans, ans)
         return elemwiseMul(g, mzz)
      end
   })
   module.gradient("sinh", {
      function(g, ans, x)
         return torch.cosh(x)
      end
   })
   module.gradient("cosh", {
      function(g, ans, x)
         return torch.sinh(x)
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
   module.gradient("clamp", {
      function(g, ans, x, minVal, maxVal)
         -- NOTE: could do a casting and then multiply for 2nd order divs. This is more efficient for now.
         local mask = torch.typeAs(torch.eq(torch.ne(ans,minVal),torch.ne(ans,maxVal)), g)
         return torch.cmul(g, mask)
      end,
      function(g, ans, x, minVal, maxVal)
         error("Gradient not implemented w.r.t. min and max values of torch.clamp")
      end,
      function(g, ans, x, minVal, maxVal)
         error("Gradient not implemented w.r.t. min and max values of torch.clamp")
      end
   })
   module.gradient("contiguous", {
      function(g,ans,x)
         return g
      end
   })
   module.gradient("cat", functions.catGradient)
   module.gradient("expand", expandGradients)
   module.gradient("expandAs", expandAsGradients)
   module.gradient("view", viewGradients)
   module.gradient("viewAs", viewAsGradients)
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
         return util.selectSliceCopy(g, x, dim, index)
      end
   })
   module.gradient("index", {
      function(g, ans, x,dim,index)
         return util.indexAdd(g, x, dim, index)
      end
   })
   module.gradient("narrow", {
      function(g, ans, x,dim,index,size)
         return util.narrowSliceCopy(g, x, dim, index, size)
      end
   })
   module.gradient("reshape", {
      function(g, ans, x, ...)
         return torch.viewAs(g, x)
      end,
      function(g, ans, x, ...) return nil end,
      function(g, ans, x, ...) return nil end,
      function(g, ans, x, ...) return nil end,
      function(g, ans, x, ...) return nil end, -- 4D is enough. Beyond that have to use LongTensor for shape

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
   module.gradient("transpose", {
      function(g, ans, x, d1, d2)
         return torch.transpose(g, d1, d2)
      end,
      function(g, ans, x, d1, d2)
         return nil
      end,
      function(g, ans, x, d1, d2)
         return nil
      end
   })
   module.gradient("t", {
      function(g, ans, x)
         return torch.t(g)
      end,
   })
   module.gradient("long", {
      function(g, ans, x)
         return torch.typeAs(g, x)
      end
   })
   module.gradient("typeAs", {
      function(g, ans, x, y)
         return torch.typeAs(g, x)
      end,
      function(g, ans, x, y)
         return nil
      end
   })

   module.gradient("repeatTensor", {
      function(g, ans, x, ...)
         local Dg = torch.nDimension(g)
         local Dx = torch.nDimension(x)
         for i=Dx,1,-1 do
            local D = torch.nDimension(g)
            local c = util.cat(torch.split(g,torch.size(x,i), Dg-Dx+i), D+1)
            g = torch.squeeze(torch.sum(c,D+1))
         end
         for i=1,Dg-Dx do
            g = torch.squeeze(torch.sum(g,1))
         end
         return g
      end,
      function(g, ans, ...) return nil end,
      function(g, ans, ...) return nil end,
      function(g, ans, ...) return nil end,
      function(g, ans, ...) return nil end,
      function(g, ans, ...) return nil end, -- five dimensions should be enough
   })
   module.gradient("squeeze", {
      function(g, ans, x)
         return torch.viewAs(g, x)
      end
   })
   -- module.gradient("split", {
   --    function(g, ans, x, size, dim)
   --       -- TODO: untested, not sure if we support table output
   --       return torch.cat(g, dim)
   --    end,
   --    function(g, ans, x, size, dim) return nil end,
   --    function(g, ans, x, size, dim) return nil end,

   -- })

   -- Zero gradients
   module.gradient("lt", zeroGradient())
   module.gradient("le", zeroGradient())
   module.gradient("gt", zeroGradient())
   module.gradient("ge", zeroGradient())
   module.gradient("eq", zeroGradient())
   module.gradient("ne", zeroGradient())
   module.gradient("all", zeroGradient())
   module.gradient("any", zeroGradient())
   module.gradient("floor", zeroGradient())
   module.gradient("ceil", zeroGradient())
   module.gradient("round", zeroGradient())
   module.gradient("sign", zeroGradient())

   module.initializer("new", "bernoulli", "uniform", "normal", "random", "zeros", "zero", "eye", "ones")
   module.static("size", "isTensor", "nDimension", "nElement", "isSameSizeAs", "setmetatable", "getmetatable", "type")

   module.ignore("typename")
   module.dynamic("split")

   module.defaultUnsupported()
end)


overload.module("Value", Value, function(module)
   for k, v in pairs(operators) do
      module.operator(k, v)
   end
   module.gradient("__internal_get", functions.get)
   module.gradient("__internal_set", functions.set)
end)

overload.module("DirectNode", DirectNode, function(module)
   for k, v in pairs(operators) do
      module.operator(k, v)
   end
   module.gradient("__internal_get", functions.get)
   module.gradient("__internal_set", functions.set)
end)

overload.module("util", util, function(module)
   module.gradient("sigmoid", {
      function(g, ans, x)
         local p = torch.cmul(1 - ans, ans)
         return torch.cmul(g, p)
      end
   })
   module.gradient("fill", {
      function(g, ans, template, x)
         return nil
      end,
      function(g, ans, template, x)
         return torch.sum(g)
      end,
   })
   module.gradient("selectSliceCopy", {
      function(g, ans, x, template, dim, index)
         return torch.select(g, dim, index)
      end,
      function(g, ans, x, template, dim, index) return nil end,
      function(g, ans, x, template, dim, index) return nil end,
      function(g, ans, x, template, dim, index) return nil end,
   })
   module.gradient("narrowSliceCopy", {
      function(g, ans, x, template, dim, index, size)
         return torch.narrow(g, dim, index, size)
      end,
      function(g, ans, x, template, dim, index, size) return nil end,
      function(g, ans, x, template, dim, index, size) return nil end,
      function(g, ans, x, template, dim, index, size) return nil end,
      function(g, ans, x, template, dim, index, size) return nil end,
   })
   module.gradient("indexAdd", {
      function(g, ans, x, template, dim, index)
         return torch.index(g, dim, index)
      end,
      function(g, ans, x, template, dim, index) return nil end,
      function(g, ans, x, template, dim, index) return nil end,
      function(g, ans, x, template, dim, index) return nil end,
   })
   module.gradient("makeContiguous", zeroGradient())
   module.gradient("cat", functions.catGradient)
   module.static("lt")
   module.static("le")
   module.static("gt")
   module.static("ge")
   module.static("eq")
   module.initializer("newTensorLike", "zerosLike")
end)


