local functionalize = require('autograd.nnwrapper').functionalize
local nn = functionalize('nn')

local softMax = nn.SoftMax()

return function(opt, params)
  local opt = opt or {}
  local params = params or {}

  local layerNormalization = opt.layerNormalization or false
  local hiddenFeatures = opt.hiddenFeatures or 10
  local subjectFeatures = opt.subjectFeatures or 15
  local subjectChoices = opt.subjectChoices or 20

  p = {W_att_subject = torch.zeros(1, 1, subjectFeatures),
       W_att_h = torch.zeros(hiddenFeatures, subjectChoices),
       b_att = torch.zeros(1, subjectChoices)}
  table.insert(params, p)

  local soft_attention = function(params, subject, h)
    -- Return soft attention over subject given hidden state.
    -- Deterministic soft attention of Show, Attend, and Tell by Xu et al.
    local batchSize = torch.size(subject, 1)
    local subjectFeatures = torch.size(subject, 2)
    local subjectChoices = torch.size(subject, 3)
    -- Activations for each subject choice and hidden state.
    local W_subject = torch.contiguous(torch.expand(params.W_att_subject, batchSize, 1, subjectFeatures))
    local subject_logit = torch.contiguous(torch.squeeze(torch.bmm(W_subject, subject), 2))
    local hidden_logit = h * params.W_att_h
    -- Focus distribution over subject choices.
    local focus_logit = subject_logit + hidden_logit
    if layerNormalization then
      focus_logit = layer_norm({gain = params.focus_ln_gain, bias = params.focus_ln_bias}, focus_logit)
    else
      focus_logit = focus_logit + torch.expand(params.b_att, batchSize, subjectChoices)
    end
    local focus = torch.contiguous(softMax(focus_logit))
    -- Attend to choice in expectation.
    local expanded_focus = torch.expand(torch.view(focus, batchSize, 1, subjectChoices), torch.size(subject))
    local attention = torch.contiguous(torch.squeeze(torch.sum(torch.cmul(subject, expanded_focus), 3), 3))
    return attention, focus
  end
  return params, soft_attention
end
