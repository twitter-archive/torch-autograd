-- utils
local util = require 'autograd.util'

-- standard loss functions
local loss = {}

function loss.logMultiNomial(out, target)
   local lsm = util.logSoftMax(out)
   return -torch.sum(torch.cmul(lsm, target))
end

return loss
