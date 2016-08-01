local util = require 'autograd.util'
return function(opt, params)
  local opt = opt or {}
  local params = params or {}

  local nOutputs = opt.nOutputs or 10
  p = {gain = torch.ones(1, nOutputs),
       bias = torch.zeros(1, nOutputs)}
  table.insert(params, p)

  local function layer_norm(params, x, eps)
    local eps = eps or 1e-5
    local n = torch.size(x,2)
    local mean = torch.expand(torch.mean(x, 2), torch.size(x))
    local x_centered = x - mean
    local std = torch.expand(torch.sqrt(torch.sum(torch.cmul(x_centered, x_centered) / n, 2)) + eps, torch.size(x))
    local x_normed = torch.cdiv(x_centered, std)
    local gain = torch.expand(params.gain, torch.size(x))
    local bias = torch.expand(params.bias, torch.size(x))
    local x_corrected = torch.cmul(x_normed, gain) + bias
    return x_corrected
  end
  return params, layer_norm
end
