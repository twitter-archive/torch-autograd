require 'pprint'
require 'autograd'

function fn_dot(A)
	B = A*A
	return torch.sum(B)
end
A = torch.DoubleTensor(3,3):fill(2)
print(fn_dot(A))

local dfn_dot = grad(fn_dot)
local A = torch.DoubleTensor(3,3):fill(2)
print(dfn_dot(A))