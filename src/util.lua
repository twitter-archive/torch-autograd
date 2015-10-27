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
   local max = torch.max(array)
   return torch.log(torch.sum(torch.exp(array-max))) + max
end

function util.logSoftMax(array)
   return array - util.logSumExp(array)
end

function util.sigmoid(array,p)
   p = p or 0
   return torch.pow(torch.exp(-array) + 1, -1) * (1-p*2) + p
end

function util.lookup(tble, indexes)
   local indexSize = indexes:size():totable()
   local rows = torch.index(tble, 1, indexes:view(-1):long())
   table.insert(indexSize, rows:size(2))
   return torch.view(rows, unpack(indexSize))
end

local fmt = getmetatable(torch.FloatTensor)
local dmt = getmetatable(torch.DoubleTensor)
local cmt = getmetatable(torch.CudaTensor) or fmt
function util.isTensor(t)
   local qmt = getmetatable(t)
   return qmt == fmt or qmt == dmt or qmt == cmt
end

return util
