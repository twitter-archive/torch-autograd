local DirectNode = require 'autograd.runtime.direct.DirectNode'
local DirectTape = require 'autograd.runtime.direct.DirectTape'

local function create(fn, opt)
   return function(...)
      if opt.withForward and opt.withGradients then
         return DirectTape.grad(fn, opt.argnum, nil, ...)
      elseif opt.withForward then
         return fn(...)
      elseif opt.withGradients then
         local args = {...}
         local partialGrad = table.remove(args, #args)
         return DirectTape.grad(fn, opt.argnum, partialGrad, table.unpack(args))
      end
   end
end

return {
   create = create
}
