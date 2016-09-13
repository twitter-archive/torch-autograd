local util = require 'autograd.util'
return function(opt, params)
  local opt = opt or {}
  local params = params or {}

  local nOutputs = opt.nOutputs or 10
  local p = {gain = torch.ones(1, nOutputs),
             bias = torch.zeros(1, nOutputs)}
  table.insert(params, p)

  local function layer_norm(params, x, eps)
    --[[ Layer Normalization of Ba, Kiros, and Hinton (https://arxiv.org/abs/1607.06450)

    Normalizes activations x at a layer by their mean and std.

    Parameters:
    * `params` - Gain and bias parameters to adjust normalized output.
    * `x` - ([batch, nOutputs]) tensor to be normalized.
    * `eps` - Small constant to avoid divide by zero for small std.

    Returns:
    * `x_corrected` - ([batch,] nOutputs]) layer normalized tensor.
    --]]
    local p = params[1] or params
    local eps = eps or 1e-5
    local x_in = x
    if torch.nDimension(x) == 1 then
      x_in = torch.view(x, 1, torch.size(x, 1))
    end
    local n = torch.size(x_in,2)
    local mean = torch.expand(torch.mean(x_in, 2), torch.size(x_in))
    local x_centered = x_in - mean
    local std = torch.expand(torch.sqrt(torch.sum(torch.cmul(x_centered, x_centered) / n, 2) + eps), torch.size(x_in))
    local x_normed = torch.cdiv(x_centered, std)
    local gain = torch.expand(p.gain, torch.size(x_in))
    local bias = torch.expand(p.bias, torch.size(x_in))
    local x_corrected = torch.view(torch.cmul(x_normed, gain) + bias, torch.size(x))
    return x_corrected
  end
  return layer_norm, params
end
