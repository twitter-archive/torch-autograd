-- Tester:
local totem = require 'totem'
local autograd = require 'autograd'
local gradcheck = require 'autograd.gradcheck'
local tester = totem.Tester()

-- List of tests:
local tests = {

   DivZero = function()
      -- Parameters:
      local W = torch.Tensor(32,100):fill(.5)
      local x = torch.Tensor(100):fill(.5)

      -- Function:
      local func = function(inputs)
         return torch.sum(torch.div(inputs.W * inputs.x, 0))
      end

      -- Grads:
      local sawHook = 0
      local dFunc = autograd(func, nil, {
         debugHook = function(debugger, msg)
            debugger.generateDot("stuff.dot")
            sawHook = sawHook + 1
         end
      })

      -- Compute func and grads:
      local pred = func({W=W, x=x})
      local grads = dFunc({W=W, x=x})

      -- Tests:
      tester:asserteq(sawHook, 5, 'debugHook should have tripped')
   end,
}

-- Run tests:
tester:add(tests):run()
