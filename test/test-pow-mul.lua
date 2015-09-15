require 'pprint'
require 'autograd'


function fn_pow_mul(A)
	print(A)
	B = torch.pow(A,2.0)
	print(B)
	D = torch.cmul(B,A)
	return D
end
A = torch.DoubleTensor(3,3):fill(2)
print(fn_pow_mul(A))

local dfn_pow_mul = grad(fn_pow_mul)
local A = torch.DoubleTensor(3,3):fill(2)
print(A)
print(dfn_pow_mul(A))