local DirectNode = require 'autograd.runtime.direct.DirectNode'
local DirectTape = require 'autograd.runtime.direct.DirectTape'
local Profiler = require 'autograd.runtime.direct.Profiler'

local function create(fn, opt)
   local pf = nil
   if opt.profile ~= 'off' then
      pf = Profiler.new()
   end
   return function(...)
      if pf ~= nil and math.fmod(pf.times + 1, opt.profileReportFrequency) == 0 then
         pf:printReport(opt.profile)
      end
      if opt.withForward and opt.withGradients then
         return DirectTape.grad(fn, opt.argnum, nil, pf, ...)
      elseif opt.withForward then
         return fn(...)
      elseif opt.withGradients then
         local args = {...}
         local partialGrad = table.remove(args, #args)
         return DirectTape.grad(fn, opt.argnum, partialGrad, nil, table.unpack(args))
      end
   end
end

return {
   create = create
}
