-- utils
local util = require 'autograd.util'

-- standard loss functions
local loss = {}

function loss.logistic(out, target)
   local lsm = util.logSoftMax(out)
   return -torch.sum(torch.cmul(lsm, target))
end

function loss.leastSquares(out, target)
   local diffs = out - target
   local sq = torch.cmul(diffs, diffs)
   return torch.sum(sq)
end

return loss
