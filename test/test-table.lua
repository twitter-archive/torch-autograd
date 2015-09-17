require 'autograd'

function fn_dot_tanh(params, x)
	local W2 = x * params.W[1] + params.B[1]
	local W3 = torch.tanh(W2)
	local W4 = W3 * params.W[2] + params.B[2]
	local out = torch.sum(W4)
	return out
end

the_fn = fn_dot_tanh

x = torch.FloatTensor(1,32):fill(0.5)
W1 = torch.FloatTensor(32, 100):fill(0.5)
B1 = torch.FloatTensor(100):fill(0.5)
W2 = torch.FloatTensor(100, 100):fill(0.5)
B2 = torch.FloatTensor(100):fill(0.5)
params = {
	W = {W1, W2}, 
	B = {B1, B2},
}
print(the_fn(params,x))

dthe_fn = grad(the_fn)
out = dthe_fn(params,x)
print(out)
print(out.W[1][1][1])
print(out.B[1][1][1])