-- autograd native modules
local module = {
  LayerNormalization = require 'autograd.module.LayerNormalization',
  MaskedBatchNormalization = require 'autograd.module.MaskedBatchNormalization',
  SoftAttention = require 'autograd.module.SoftAttention'
}

return module
