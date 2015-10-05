-- Utilities
local util = {}

function util.oneHot(labels, n)
   --[[
   Assume labels is a 1D tensor of contiguous class IDs, starting at 1.
   Turn it into a 2D tensor of size labels:size(1) x nUniqueLabels

   This is a pretty dumb function, assumes your labels are nice.
   ]]
   local n = n or labels:max()
   local nLabels = labels:size(1)
   local out = torch.FloatTensor(nLabels, n):fill(0)
   for i=1,nLabels do
      out[i][labels[i]] = 1.0
   end
   return out
end

-- Helpers:
function util.logMultiNomialLoss(out, target) return -torch.sum(torch.cmul(out,target)) end
function util.logSumExp(array) return torch.log(torch.sum(torch.exp(array))) end
function util.logSoftMax(array) return array - util.logSumExp(array) end

return util
