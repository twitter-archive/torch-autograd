require 'autograd'

function fn_sum(A)
	E = torch.pow(A,3)
	return E
end

A = torch.FloatTensor(3,3):fill(2)
print(fn_sum(A))

dfn_sum = grad(fn_sum)
Q = torch.FloatTensor(3,3):fill(2)
out1 = dfn_sum(Q)
out2 = dfn_sum(Q)