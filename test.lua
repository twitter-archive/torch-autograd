require 'pprint'
require 'autograd'


function scale(A)
	return adtorch.mul(A,3.0)
end

dscale = grad(scale)

A = torch.FloatTensor(3,3):fill(2)

print(scale(A))
print(dscale(A))

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