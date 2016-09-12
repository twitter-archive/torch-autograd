local functionalize = require('autograd.nnwrapper').functionalize
local nn = functionalize('nn')
local LayerNorm = require 'autograd.module.LayerNormalization'

local softMax = nn.SoftMax()

return function(opt, params)
  local opt = opt or {}
  local params = params or {}

  local layerNormalization = opt.layerNormalization or false
  local hiddenFeatures = opt.hiddenFeatures or 10
  local subjectFeatures = opt.subjectFeatures or 15
  local subjectChoices = opt.subjectChoices or 20

  local p = {W_att_subject = torch.zeros(1, 1, subjectFeatures),
             W_att_h = torch.zeros(hiddenFeatures, subjectChoices),
             b_att = torch.zeros(1, subjectChoices)}

  if layerNormalization then
    local focus_ln_params = LayerNorm({nOutputs = subjectChoices})
    p.focus_ln_gain = focus_ln_params.gain
    p.focus_ln_bias = focus_ln_params.bias
    p.b_att = nil
  end
  table.insert(params, p)

  local soft_attention = function(params, subject, h)
    --[[ Soft attention over subject given hidden state.

    Deterministic soft attention of Show, Attend, and Tell by Xu et al. (http://arxiv.org/abs/1502.03044)

    Parameters:
    * `params`  - Weights to combine subject and hidden features to score choices.
    * `subject` - ([batch,] subjectFeatures, subjectChoices) tensor.
    * `h`       - ([batch,] hiddenFeatures) tensor.

    Returns:
    * `attention` - ([batch,], subjectFeatures) tensor that is the expectation of the attended subject vector.
    * `focus`     - ([batch,], subjectChoices) tensor that is the probability of selecting any given subject choice.
    --]]
    local p = params[1] or params
    local subject_in = subject
    local h_in = h
    if torch.nDimension(subject) == 2 then
      subject_in = torch.view(subject, 1, torch.size(subject, 1), torch.size(subject, 2))
    end
    if torch.nDimension(h) == 1 then
      h_in = torch.view(h, 1, torch.size(h, 1))
    end
    local batchSize = torch.size(subject_in, 1)
    local subjectFeatures = torch.size(subject_in, 2)
    local subjectChoices = torch.size(subject_in, 3)
    -- Activations for each subject choice and hidden state.
    local W_subject = torch.expand(p.W_att_subject, batchSize, 1, subjectFeatures)
    local subject_logit = torch.squeeze(torch.bmm(W_subject, subject_in), 2)
    local hidden_logit = h_in * p.W_att_h
    -- Focus distribution over subject choices.
    local focus_logit = subject_logit + hidden_logit
    if layerNormalization then
      focus_logit = layer_norm({gain = p.focus_ln_gain, bias = p.focus_ln_bias}, focus_logit)
    else
      focus_logit = focus_logit + torch.expand(p.b_att, batchSize, subjectChoices)
    end
    local focus = softMax(focus_logit)
    -- Attend to choice in expectation.
    local expanded_focus = torch.expand(torch.view(focus, batchSize, 1, subjectChoices), torch.size(subject_in))
    local attention = torch.squeeze(torch.sum(torch.cmul(subject_in, expanded_focus), 3), 3)
    if torch.nDimension(subject) == 2 then
      attention = torch.squeeze(attention, 1)
      focus = torch.squeeze(focus, 1)
    end
    return attention, focus
  end
  return soft_attention, params
end
