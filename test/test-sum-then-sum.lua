require 'autograd'

function fn_sumsum(A)
	local b = torch.sum(A)
	return torch.sum(A+b)
end

A = torch.FloatTensor(3,3):fill(1e-3)
B = torch.FloatTensor(3,3):fill(1e-3)
print(fn_sumsum(A))

dfn_sumsum = grad(fn_sumsum)
Q = torch.FloatTensor(3,3):fill(1e-3)
out = dfn_sumsum(Q, B)
print(out)