require 'pprint'
require 'autograd'


function fn_pow(A)
	B = torch.pow(A,3.0) + A
	return B
end

A = torch.DoubleTensor(3,3):fill(2)
dfn_pow = grad(fn_pow)
Q = torch.DoubleTensor(3,3):fill(2)
print(dfn_pow(Q))