require 'autograd'

-- print = require 'print'
local function logMultiNomialLoss(out, target) return -torch.sum(torch.cmul(out,target)) end
local function logsumexp(array) return torch.log(torch.sum(torch.exp(array))) end
local function logsoftmax(array) return array - logsumexp(array) end
local function uniform(min, max, h, w) return torch.mul(torch.FloatTensor():rand(h,w), (max-min)) + min end

inputSize = 32

function neural_net(params, input, target)
	local W2 = torch.tanh(input * params.W[1] + params.B[1])
	local W3 = torch.tanh(W2 * params.W[2] + params.B[2])
	local W4 = torch.tanh(W3 * params.W[3] + params.B[3])
	local out = logsoftmax(W4)
	local loss = logMultiNomialLoss(out, target)
	return loss
end


-- Define our parameters
-- [-1/sqrt(#output), 1/sqrt(#output)]
torch.manualSeed(0)
-- x = torch.FloatTensor(1,inputSize):fill(0.5)
-- W1 = torch.FloatTensor(inputSize, 100)
W1 = uniform(-1/math.sqrt(100),1/math.sqrt(100), inputSize, 100)
B1 = torch.FloatTensor(100):fill(0)
W2 = uniform(-1/math.sqrt(100),1/math.sqrt(100), 100, 100)
B2 = torch.FloatTensor(100):fill(0.0)
W3 = uniform(-1/math.sqrt(10),1/math.sqrt(10), 100, 10)
B3 = torch.FloatTensor(10):fill(0.0)

local params = {
	W = {W1, W2, W3}, 
	B = {B1, B2, B3},
}

-- Check that the neural net runs forward properly

-- Get some gradients
local dneural_net = grad(neural_net)
local x = uniform(-1,1,1,inputSize)
local y = torch.FloatTensor(1,10):zero()
y[1][3] = 1
print(y)


print("1===========================")
print(params.W[1].name)
out = dneural_net(params,x,y)
print("2===========================")
print(params.W[1].name)
print("3===========================")
out = dneural_net(params,x,y)
print(params.W[1].name)
print("4===========================")
print(out)

-- -- print(tape)

-- -- os.exit()
-- -- -- momgradW = [w*0.0 for w in weights]
-- -- -- for epoch in xrange(1000):
-- -- --   l.append(loss(weights, inputs, values, hypers))
-- -- --   gradW = gradWfun(weights, inputs, values, hypers)

-- -- --   for i in xrange(len(gradW)):
-- -- --     gradW[i] = np.clip(gradW[i], -5.0, 5.0)
-- -- --     momgradW[i] = 0.9 * momgradW[i] + 0.1*gradW[i]
-- -- --     weights[i]  = weights[i] - 0.01 * momgradW[i]

-- -- Do an SGD step
print(out.W[1])
print(torch.sum(out.W[1]))
print(torch.sum(out.B[1]))

print(torch.sum(x))
