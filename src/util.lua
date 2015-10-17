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
   local out = labels.new(nLabels, n):fill(0)
   for i=1,nLabels do
      out[i][labels[i]] = 1.0
   end
   return out
end

-- Helpers:
function util.logMultinomialLoss(out, target)
   return -torch.sum(torch.cmul(out,target))
end

function util.logSumExp(array)
   local max = 0
   if torch.typename(array) then -- TODO: fix autograd (missing support for max)
      max = array:max()
   end
   return torch.log(torch.sum(torch.exp(array-max))) + max
end

function util.logSoftMax(array)
   return array - util.logSumExp(array)
end

function util.sigmoid(array,p)
   p = p or 0
   return torch.pow(torch.exp(-array) + 1, -1) * (1-p*2) + p
end

return util
