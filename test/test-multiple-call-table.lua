require 'autograd'

function fn_sum(A)
	B = torch.cmul(A.data, A.stuff)
	E = torch.pow(B,3)
	return E
end

A = {
	data=torch.FloatTensor(3,3):fill(2),
	stuff=torch.FloatTensor(3,3):fill(3)
}
print(fn_sum(A))

dfn_sum = grad(fn_sum)
Q = {data=torch.FloatTensor(3,3):fill(2)}
print("Once")
out1 = dfn_sum(A)
print("Twice")
out2 = dfn_sum(A)