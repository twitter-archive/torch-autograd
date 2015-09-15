require 'pprint'
require 'autograd'


function fn_scale(A)
	B = torch.mul(A,3.0) + A
	return B
end

A = torch.DoubleTensor(3,3):fill(2)
print(fn_scale(A))

dfn_scale = grad(fn_scale)
Q = torch.DoubleTensor(3,3):fill(2)
print(dfn_scale(Q))


