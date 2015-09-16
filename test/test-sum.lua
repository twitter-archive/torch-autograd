require 'autograd'

function fn_sum(A)
	B = torch.pow(A, 2.0)
	return torch.sum(B)
end

A = torch.DoubleTensor(3,3):fill(2)
print(fn_sum(A))

dfn_sum = grad(fn_sum)
Q = torch.DoubleTensor(3,3):fill(2)
print(dfn_sum(Q))