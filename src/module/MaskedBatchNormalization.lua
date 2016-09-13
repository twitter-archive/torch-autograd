local util = require 'autograd.util'
return function(opt, params)
  local opt = opt or {}
  local params = params or {}

  local nOutputs = opt.nOutputs or 10
  local momentum = opt.momentum or 0.1

  local batchNormState = {momentum = momentum, train = 1,
                         running_mean = torch.zeros(1, nOutputs),
                         running_std = torch.ones(1, nOutputs)}

  -- initializing gain to < 1 is recommended for LSTM batch norm.
  local p = {gain = torch.zeros(1, nOutputs):fill(0.1),
             bias = torch.zeros(1, nOutputs)}
  table.insert(params, p)

  local function masked_batch_norm(params, x, mask, state, eps)
    --[[ Masked batch normalization for minibatches with variable length sequences.

    Based on sequence batch norm from Batch Normalized Recurrent Neural Networks by Laurent et al.
    (http://arxiv.org/abs/1510.01378)

    Parameters:
    * `params` - Gain and bias parameters to adjust normalized output.
    * `x` - ([batch, [time,], nOutputs]) tensor to be normalized.
    * `mask` - Tensor with the same size as x that is 1 where x is valid and 0 otherwise.
    * `state` - Running mean and std estimates, momentum for estimates, and train flag.
    * `eps` - Small constant to avoid divide by zero for small std.

    Returns:
    * `x_corrected` - ([batch, [time,], nOutputs]) batch normalized tensor.
    --]]
    local p = params[1] or params
    local eps = eps or 1e-5
    local train = state.train or 1
    local momentum = (state.momentum or 0.1) * train -- kill state updates during evaluation
    local x_in = x
    local mask_in = mask
    if torch.nDimension(x) == 3 then -- collapse batch and time dimensions
      x_in = torch.view(x, -1, torch.size(x, 3))
      mask_in = torch.view(mask, -1, torch.size(mask, 3))
    elseif torch.nDimension(x) == 1 then -- expand batch dimension
      x_in = torch.view(x, 1, torch.size(x, 1))
      mask_in = torch.view(mask, 1, torch.size(mask, 1))
    end
    local n = torch.sum(mask)
    mask_in = torch.expand(mask_in, torch.size(x_in))
    local x_masked = torch.cmul(x_in, mask_in)
    local mean = torch.sum(x_masked / n, 1)
    state.running_mean = momentum * mean + (1 - momentum) * state.running_mean
    local x_centered = torch.cmul(x_masked - torch.expand(state.running_std, torch.size(x_in)), mask_in)
    local var = torch.sum(torch.cmul(x_centered, x_centered) / n, 1) + eps
    local std = torch.sqrt(var)
    state.running_std = momentum * std + (1 - momentum) * state.running_std
    local x_normed = torch.cdiv(x_centered, torch.expand(state.running_std, torch.size(x_in)))
    local gain = torch.expand(p.gain, torch.size(x_in))
    local bias = torch.expand(p.bias, torch.size(x_in))
    local x_corrected = torch.view(torch.cmul(x_normed, gain) + bias, torch.size(x))
    return x_corrected
  end
  return masked_batch_norm, params, batchNormState
end
