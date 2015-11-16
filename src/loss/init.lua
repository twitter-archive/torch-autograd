-- utils
local util = require 'autograd.util'

-- standard loss functions
local loss = {}

-- each loss function takes a raw output of a network (last hidden layer)
-- and produces the loss, plus the transformed ouptut (in the case of a
-- binary cross entropy loss, the output goes through a sigmoid)

function loss.logMultinomialLoss(out, target)
   return -torch.sum(torch.cmul(out,target))
end

function loss.logBCELoss(out, target, p)
   if p then
      out = out + p
   end

   return -torch.sum(torch.cmul(target, torch.log(out)) + torch.cmul(-target+1, torch.log(-out+1)))
end

function loss.crossEntropy(out, target)
   local yhat = util.logSoftMax(out)
   return -torch.sum(torch.cmul(yhat, target)), yhat
end

function loss.binaryCrossEntropy(out, target)
   local p = 1e-6
   local yhat = util.sigmoid(out) * (1-p*2) + p
   return - torch.sum( torch.cmul(target, torch.log(yhat)) + torch.cmul((-target + 1), torch.log(-yhat + 1)) ), yhat
end

function loss.leastSquares(out, target)
   local yhat = out
   local diffs = out - target
   local sq = torch.cmul(diffs, diffs)
   return torch.sum(sq), yhat
end

function loss.margin(out, target, margin)
   margin = margin or 1
   local preds1 = out[torch.eq(target,1)]
   local preds0 = out[torch.eq(target,0)]
   local np1 = preds1:size(1)
   local np0 = preds0:size(1)
   local diffs = torch.expand( torch.view(preds1, np1, 1), np1, np0 ) - torch.expand( torch.view(preds0, 1, np0), np1, np0 )
   diffs = -diffs + margin
   local max0s = diffs[ torch.gt(diffs, 0) ]
   local loss = torch.sum(max0s)
   return loss, out
end

return loss
