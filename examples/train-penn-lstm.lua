-- Libs
local grad = require 'autograd'
local util = require 'autograd.util'

-- Load in PENN Treebank dataset
local trainData, valData, testData, dict = require('./get-penn.lua')()

print('loaded data: ', {
   train = trainData,
   validation = valData,
   test = testData,
   nTokens = #dict.id2word,
})
