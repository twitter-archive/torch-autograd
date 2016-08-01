local util = require 'autograd.util'
-- masked batch norm for minibatches with variable length sequences
return function(opt, params)
  local opt = opt or {}
  local params = params or {}

  local nOutputs = opt.nOutputs or 10
  local momentum = opt.momentum or 0.1

  batchNormState = {momentum = momentum, train = true,
                    running_mean = torch.zeros(1, nOutputs),
                    running_std = torch.zeros(1, nOutputs)}

  -- initializing gain to < 1 is recommended for LSTM batch norm.
  p = {gain = torch.zeros(1, nOutputs):fill(0.1),
       bias = torch.zeros(1, nOutputs)}
  table.insert(params, p)

  local function masked_batch_norm(params, x, mask, state, eps)
    local eps = eps or 1e-5
    local momentum = state.momentum or 0.1
    local train = state.train or 1
    local n = torch.sum(mask)
    local mask = torch.expand(mask, torch.size(x))
    local x_masked = torch.cmul(x, mask)
    local x_normed
    local mean = torch.sum(x_masked / n, 1)
    mean = torch.mul(mean, train) + torch.mul(state.running_mean, 1 - train)
    local combined_mean = momentum * mean + (1 - momentum) * state.running_mean
    state.running_mean = combined_mean -- NOTE: Assignment doesn't work in direct mode.
    local x_centered = torch.cmul(x_masked - torch.expand(combined_mean, torch.size(x)), mask)
    local var = torch.sum(torch.cmul(x_centered, x_centered) / n, 1) + eps
    local std = torch.sqrt(var)
    std = torch.mul(std, train) + torch.mul(state.running_std, 1 - train)
    local combined_std = momentum * std + (1 - momentum) * state.running_std
    state.running_std = combined_std
    x_normed = torch.cdiv(x_centered, torch.expand(combined_std, torch.size(x)))
    local gain = torch.expand(params.gain, torch.size(x))
    local bias = torch.expand(params.bias, torch.size(x))
    local x_corrected = torch.cmul(x_normed, gain) + bias
    return x_corrected
  end
  return params, masked_batch_norm
end
