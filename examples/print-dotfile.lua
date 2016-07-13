print("This example will only fully work on Mac OS X, because we use Safari to view the resulting SVG file of the compute graph")

require 'torch'
autograd = require 'autograd'


-- Just a standard MLP, for demonstration purposes
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
local inputSize = 1024
local classes = {0,1,2,3,4,5,6,7,8,9}
-- What model to train:
local predict,f,params

-- Define our neural net
function predict(params, input, target)
   local h1 = torch.tanh(input * params.W[1] + params.B[1])
   local h2 = torch.tanh(h1 * params.W[2] + params.B[2])
   local h3 = h2 * params.W[3] + params.B[3]
   local out = autograd.util.logSoftMax(h3)
   return out
end

-- Define our training loss
function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = autograd.loss.logMultinomialLoss(prediction, target)
   return loss, prediction
end

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
local W1 = torch.Tensor(inputSize,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
local B1 = torch.Tensor(50):fill(0)
local W2 = torch.Tensor(50,50):uniform(-1/math.sqrt(50),1/math.sqrt(50))
local B2 = torch.Tensor(50):fill(0)
local W3 = torch.Tensor(50,#classes):uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes))
local B3 = torch.Tensor(#classes):fill(0)

-- Trainable parameters:
params = {
   W = {W1, W2, W3},
   B = {B1, B2, B3},
}

input = torch.randn(1,inputSize)
target = torch.zeros(1,#classes)
target[1][3] = 1
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------

-- Get our function, just the forward pass, along with the DOT file showing the graph
fileName = "/tmp/graph.dot"
g = autograd(f,{dotFile=fileName,optimize=true,withGradients=false,withForward=true})

-- Run the function to produce the dotfile
print(g(params,input,target))

-- Show the dotfile
os.execute('dot -O -Tsvg ' .. fileName)
os.remove(fileName)
os.execute('open -a Safari ' .. fileName ..'.svg')

