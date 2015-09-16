require 'autograd'

function fn_dot_tanh(W, x)
	local W2 = x * W
	local W3 = torch.tanh(W2)
	local out = torch.sum(W3)
	return out
end

the_fn = fn_dot_tanh

x = torch.DoubleTensor(1,32):fill(0.5)
W = torch.DoubleTensor(32, 100):fill(0.5)
-- -- print(the_fn(W,x))

dthe_fn = grad(the_fn)
print(dthe_fn(W,x))

-- print(W+W)