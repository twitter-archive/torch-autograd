torch = torch or require('torch')
adtorch = {}
class = require 'class'
require 'util'

op = {
	add = function(a,b) return a+b end,
	mult = function(a,b) return a*b end,
	div = function(a,b) return a/b end, 
	pow = function(a,b) return a^b end,
}

Node = class("Node")

function Node:__init(value, fun, args, tape)
    self.value = value
    self.fun = fun
    self.args = args
    self.tape = tape or {}
    self.tape[#self.tape+1] = self
    self.outgrad = 0.0
end

function Node:__tostring()
	return tostring(self.value)
end

function Node.__add(l,r)
	return nodeapply(op.add, l, r)
end

function Node.__mul(l,r)
	return nodeapply(op.mult, l, r)
end

function Node.__div(l,r)
	return nodeapply(op.div, l, r)
end
function Node.__pow(l,r)
	return nodeapply(op.pow, l, r)
end

-- A wrapper for a function
-- Anytime we try to apply a function to some arguments,
-- we'd like to make sure that if we're passing nodes in,
-- that we unpack the value in those nodes, apply the function
-- to the underlying value, and then wrap the value in a node
function nodeapply(fun, ...)
    local arg = {...}
    local parents = filter(isnode, arg)
    if #parents > 0 then
        local vals = map(getval,arg)
        local value = nodeapply(fun,unpack(map(getval,arg)))
        local out = Node(value, fun, arg, parents[1].tape)
        return out
    else
        return fun(unpack(arg))
    end
end


-- Step through and take the gradient
function grad(fun, argnum)
	argnum = argnum or 1
	gradfun = function(...)
		local arg = {...}
		local tape = {}

		arg[argnum] = Node(arg[argnum], nil, nil, tape)
		
		ans = fun(unpack(arg))
		if not isnode(ans) then
			return 0.0
		end
		ans.outgrad = 1.0
		local node
		for i=#ans.tape,1,-1 do
			node = ans.tape[i]
			for iarg=1,#node.args do
				this_arg = node.args[iarg]
				if isnode(this_arg) then
					gradfun = gradfuns[node.fun][iarg]
					grad_update = gradfun(node.outgrad, unpack(map(getval,node.args)))
					this_arg.outgrad = this_arg.outgrad + grad_update
				end
			end
		end
		return node.outgrad
	end
	return gradfun
end


override = { 
	"__add", "add", 
	"__mul", "mul", "cmul",
	"pow", "__pow",
	"sum", "repeatTensor",
	"dot",
	}

gradfuns = {}
gradfuns[op.add] = {
	function(g, x, y) return g end,
	function(g, x, y) return g end,
}
gradfuns[op.mult] = {
	function(g, x, y) return y*g end,
	function(g, x, y) return x*g end,
}
gradfuns[op.div] = {
	function(g, x, y) return g/y end,
	function(g, x, y) return -g*x/(y^2) end,
}
gradfuns[op.pow] = {
	function(g, x, y) return g * y * x ^ (y-1) end,
}
gradfuns[torch.add] = {
	function(g, x, y) return g end,
	function(g, x, y) return g end,
}
gradfuns[torch.cmul] = {
	function(g, x, y) return y*g end,
	function(g, x, y) return x*g end,	
}
gradfuns[torch.pow] = {
	function(g, x, y) return g * y * x ^ (y-1) end
}
gradfuns[torch.dot] = {
	function(g, A, B) return g * B end,
	function(g, A, B) return g * A end,
}

for ifn,fn_name in pairs(override) do
	local old = torch[fn_name]
	new_fn = function(...)
		return nodeapply(old, ...)
	end

	-- Make sure that torch.cmul(a,b) gets the same
	-- treatment as a:cmul(b)
	adtorch[fn_name] = new_fn
end




------------------------------------------
-- Not clear how to do in-place operations
------------------------------------------

-- tensor_types = {
-- 	'FloatTensor',
-- 	'DoubleTensor'
-- }
-- for _,tensor_type in pairs(tensor_types) do

-- 	-- Metatable for Torch Tensors.
-- 	mt = torch.getmetatable('torch.' .. tensor_type)
-- 	mt['__node'] = node_fn

-- 	-- Then, override all the methods we need to override
-- 	do
-- 		for fn_name,fn_override in pairs(override) do
-- 			local old = mt[fn_name]
-- 			mt[fn_name] = function(...)
-- 				return nodeapply(old, ...)
-- 			end

-- 			-- Make sure that torch.cmul(a,b) gets the same
-- 			-- treatment as a:cmul(b)
-- 			rawset(torch, fn_name, mt[fn_name])
-- 		end
-- 	end
-- end