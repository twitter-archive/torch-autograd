require 'pprint'
require 'autograd'


function fn_mul(A)
	B = torch.cmul(A,A)
	C = A + A
	D = C * B
	return D
end

A = torch.DoubleTensor(3,3):fill(2)
dfn_mul = grad(fn_mul)
Q = torch.DoubleTensor(3,3):fill(2)
print(dfn_mul(Q))