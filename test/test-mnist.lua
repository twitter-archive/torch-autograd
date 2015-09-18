require 'autograd'
optim = require 'optim'
require 'util'
local function logMultiNomialLoss(out, target) return -torch.sum(torch.cmul(out,target)) end
local function logsumexp(array) return torch.log(torch.sum(torch.exp(array))) end
local function logsoftmax(array) return array - logsumexp(array) end
local function uniform(min, max, h, w) return torch.mul(torch.FloatTensor():rand(h,w), (max-min)) + min end

-- Load in MNIST
trainData, testData, classes = require('./setup-data.lua')()
trainData.y = one_hot(trainData.y)
testData.y = one_hot(testData.y)
inputSize = trainData.x[1]:nElement()
confusionMatrix = optim.ConfusionMatrix(classes)

-- Define our neural net
function neural_net(params, input, target, return_prediction)
	local W2 = torch.tanh(input * params.W[1] + params.B[1])
	local W3 = torch.tanh(W2 * params.W[2] + params.B[2])
	local W4 = W3 * params.W[3] + params.B[3]
	local out = logsoftmax(W4)
	if return_prediction then
		return out
	else
		return logMultiNomialLoss(out, target)
	end
end

-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
W1 = uniform(-1/math.sqrt(100),1/math.sqrt(100), inputSize, 100)
B1 = torch.FloatTensor(100):fill(0)
W2 = uniform(-1/math.sqrt(100),1/math.sqrt(100), 100, 100)
B2 = torch.FloatTensor(100):fill(0.0)
W3 = uniform(-1/math.sqrt(#classes),1/math.sqrt(#classes), 100, #classes)
B3 = torch.FloatTensor(#classes):fill(0.0)

local params = {
	W = {W1, W2, W3}, 
	B = {B1, B2, B3},
}

-- Check that the neural net runs forward properly

require 'nn'
model = nn.Sequential()
	:add(nn.Linear(inputSize,100)):add(nn.ReLU())
	:add(nn.Linear(100,10)):add(nn.LogSoftMax())
loss = nn.ClassNLLCriterion()
w,dw = model:getParameters()
model:float()
loss:float()

-- Get some gradients
for epoch=1,100 do
	print('EPOCH #'..epoch)
	for i=1,60000 do --trainData.size do
		-- local dneural_net = grad(neural_net, 1, false)
		local x = trainData.x[i]:view(1,inputSize)
		local y = torch.view(trainData.y[i], 1, 10):clone()
		-- -- print(neural_net(params,x, y)) -- trainData.y[1]))
		-- grads = dneural_net(params,x,y)
		-- prediction = neural_net(params,x,y,true)

		-- -- Update weights and biases
		-- for i=1,#params.W do
		-- 	-- print(grads.W)
		-- 	-- params.W[i] = params.W[i] - grads.W[i] * 0.01
		-- 	-- params.B[i] = params.B[i] - grads.B[i] * 0.01
		-- 	params.W[i] = params.W[i] - grads.W[i]:clamp(-20, 20) * 0.01
		-- 	params.B[i] = params.B[i] - grads.B[i]:clamp(-20, 20) * 0.01
		-- end

		dw:zero()
		_,yi = y[1]:max(1)
		yi = yi[1]
		x = x[1]
		o = model:forward(x)
		dl_do = loss:backward(o,yi)
		model:backward(x,dl_do)
		w:add(-0.01, dw)
		confusionMatrix:add(o, yi)
		if i % 10000 == 0 then
			print(x:min())
			print(x:max())
			print(confusionMatrix)
			confusionMatrix:zero()
		end

		-- -- print(prediction)
		-- confusionMatrix:add(prediction, y)
		-- if i % 5000 == 0 then
		-- 	print(params.B[1][1])
		-- 	-- print(torch.sum(grads.W[1]))
		-- 	print(confusionMatrix)
		-- 	confusionMatrix:zero()
		-- end
	end
end

-- print(tape)

-- os.exit()
-- -- momgradW = [w*0.0 for w in weights]
-- -- for epoch in xrange(1000):
-- --   l.append(loss(weights, inputs, values, hypers))
-- --   gradW = gradWfun(weights, inputs, values, hypers)

-- --   for i in xrange(len(gradW)):
-- --     gradW[i] = np.clip(gradW[i], -5.0, 5.0)
-- --     momgradW[i] = 0.9 * momgradW[i] + 0.1*gradW[i]
-- --     weights[i]  = weights[i] - 0.01 * momgradW[i]

-- Do an SGD step
-- print(torch.sum(out.W[1]))
-- print(torch.sum(out.B[1]))

-- print(torch.sum(x))
-- print(trainData.y[1])
