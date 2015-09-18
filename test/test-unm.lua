require 'autograd'

function fn_sumsum(A)
	return torch.sum(-A)
end

A = torch.FloatTensor(3,3):fill(1e-3)
print(fn_sumsum(A))

dfn_sumsum = grad(fn_sumsum)
Q = torch.FloatTensor(3,3):fill(1e-3)
print(dfn_sumsum(Q))