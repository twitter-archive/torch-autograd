require 'pprint'
require 'autograd'


function fn_scale(A)
	B = A * 3.0 + A
	return B
end

A = torch.DoubleTensor(3,3):fill(2)
print(fn_scale(A))

dfn_scale = grad(fn_scale)
Q = torch.DoubleTensor(3,3):fill(2)
print(dfn_scale(Q))

-- function fn_pow(A)
-- 	B = torch.pow(A,3.0)
-- 	return B
-- end

-- -- A = torch.DoubleTensor(3,3):fill(2)
-- -- print(fn_pow(A))

-- dfn_pow = grad(fn_pow)
-- Q = torch.DoubleTensor(3,3):fill(2)
-- print(dfn_pow(Q))


-- print(b.__node)
-- b.__node['hello'] = 3
-- local c = 1.0
-- print(b.__node)
-- b:__node()
-- pprint(getmetatable(b)['__node'](b))

-- pprint(isnode(a))

-- pprint(tape)
-- pprint(torch.FloatTensor(3,3):fill(2):add(1))
-- pprint(torch.FloatTensor(3,3):fill(2) + 1)

-- pprint(tape)
-- pprint(torch.FloatTensor(3,3):fill(2) + 1)
-- pprint(torch.FloatTensor(3,3):fill(2):add(1))

-- pprint(tape)
-- pprint(torch.DoubleTensor(3,3):fill(2) + 1)
-- pprint(torch.DoubleTensor(3,3):fill(2):add(1))

-- pprint(tape)
-- pprint(torch.DoubleTensor(3,3):fill(2) * 2)
-- pprint(torch.DoubleTensor(3,3):fill(2):mul(1))

-- pprint(tape)
-- a = torch.DoubleTensor(3,3):fill(2)
-- b = torch.DoubleTensor(3,3):fill(2)
-- pprint(torch.cmul(a,a,b))
-- pprint(tape)
-- pprint(a:cmul(b))
-- pprint(tape)