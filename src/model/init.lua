-- standard models
local model = {
   NeuralNetwork = require 'autograd.model.NeuralNetwork',
   SpatialNetwork = require 'autograd.model.SpatialNetwork',
   RecurrentNetwork = require 'autograd.model.RecurrentNetwork',
   RecurrentLSTMNetwork = require 'autograd.model.RecurrentLSTMNetwork',
}

return model
