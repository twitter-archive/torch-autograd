-- TODO
-- Error reporting of __index failure on gradfuns

torch = torch or require('torch')
class = require 'class'
require 'util'
debug = require 'debug'
-- local STP = require "StackTracePlus"
-- debug.traceback = STP.stacktrace

local op = {
	add = function(a,b) return a+b end,
	mult = function(a,b) return a*b end,
	div = function(a,b) return a/b end, 
	pow = function(a,b) return a^b end,
}
local gradfuns = {}
local nodeapply

Node = class("Node")

function Node:__init(value, fun, args, tape)
    self.value = value
    self.fun = fun
    self.args = args or {}
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

local number_mt = {
	__add = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			return b + a
		else
			return a + b
		end
	end,
	__sub = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			return -b + a
		else
			return a - b
		end
	end,
	__mul = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			return b * a
		else
			return a * b
		end
	end,
	__div = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			-- THIS IS INSANE
			c = torch.ones(b:size())
			return torch.cdiv(c,b) * a
		else
			return a / b
		end
	end
}
debug.setmetatable(1.0, number_mt)

-- A wrapper for a function
-- Anytime we try to apply a function to some arguments,
-- we'd like to make sure that if we're passing nodes in,
-- that we unpack the value in those nodes, apply the function
-- to the underlying value, and then wrap the value in a node
nodeapply = function(fun, ...)
    local arg = {...}
    local parents = filter(isnode, arg)
    if #parents > 0 then
        local vals = map(getval,arg)
        local value = nodeapply(fun,unpack(map(getval,arg)))
        return Node(value, fun, arg, parents[1].tape)
    else
        return fun(unpack(arg))
    end
end


-- Step through and take the gradient
function grad(fun, argnum)
	argnum = argnum or 1
	local do_grad = function(...)
		local arg = {...}
		local tape = {}
		arg[argnum] = Node(arg[argnum], nil, nil, tape)
		local ans = fun(unpack(arg))
		if not isnode(ans) then
			return 0.0
		end
		ans.outgrad = 1.0
		local node

		print(map(function(t)
				if t.fun then
					return gradfuns[t.fun][1]
				else
					return nil
				end
			end,
			ans.tape))

		for i=#ans.tape,1,-1 do
			node = ans.tape[i]
			print(node.outgrad)
			for iarg=1,#node.args do
				-- print("====================================")
				-- print(node.args)
				-- print(node.outgrad)
				-- print(iarg)
				-- print("====================================")
				local this_arg = node.args[iarg]
				if isnode(this_arg) then
					local gradfun = gradfuns[node.fun][iarg+1]
					local grad_update = gradfun(node.outgrad, unpack(map(getval,node.args)))
					this_arg.outgrad = this_arg.outgrad + grad_update
				end
			end
		end
		return node.outgrad
	end
	return do_grad
end


local override = { 
	"__add", "add", 
	"__mul", "mul", "cmul",
	"pow", "__pow",
	"sum", "dot", "exp", 'tanh'
	}


local function elemwise_mult(a,b)
	if torch.isTensor(a) and torch.isTensor(b) then
		return torch.cmul(a,b)
	else
		return a*b
	end
end

local function elemwise_div(a,b)
	if torch.isTensor(a) and torch.isTensor(b) then
		return torch.cdiv(a,b)
	else
		return a/b
	end
end

gradfuns[op.add] = {
	"add",
	function(g, x, y) return g end,
	function(g, x, y) return g end,
}
gradfuns[op.mult] = {
	"mult/dot",
	function(g, A, B)
		if torch.isTensor(A) and torch.isTensor(B) then
			if B:nDimension() == 2 then
				return g*B:t()
			elseif A:nDimension() == 2 then
				return torch.ger(g, B) -- outer product
			else
				return g*B -- elemwise_mult required? what about 3D?
			end
		else
			return g*B
		end
	end,
	function (g, A, B)
		if torch.isTensor(A) and torch.isTensor(B) then
			if A:nDimension() == 2 then
				return A:t()*g
			elseif B:nDimension() == 2 then
				return torch.ger(A, g)
			else
				return g*A
			end
		else
			return g*A
		end
	end,
}
gradfuns[op.div] = {
	"div",
	function(g, x, y) return elemwise_div(g,y) end,
	function(g, x, y) return elemwise_mult(-g,elemwise_div(x,torch.pow(y,2))) end,
}
gradfuns[op.pow] = {
	"pow",
	function(g, x, y) return elemwise_mult(elemwise_mult(g,y),torch.pow(x, (y-1))) end,
}
gradfuns[torch.add] = {
	"torch.add",
	function(g, x, y) return g end,
	function(g, x, y) return g end,
}
gradfuns[torch.cmul] = {
	"torch.cmul",
	function(g, x, y) return elemwise_mult(y,g) end,
	function(g, x, y) return elemwise_mult(x,g) end,	
}
gradfuns[torch.mul] = {
	"torch.mul",
	function(g, x, y) return elemwise_mult(y,g) end,
	function(g, x, y) return elemwise_mult(x,g) end,	
}
gradfuns[torch.pow] = {
	"torch.pow",
	function(g, x, y) return elemwise_mult(elemwise_mult(g,y),torch.pow(x,y-1)) end
}
gradfuns[torch.dot] = {

}
gradfuns[torch.exp] = {
	"exp",
	function(g,x) return elemwise_mult(torch.exp(x), g) end,
}
gradfuns[torch.tanh] = {
	"tanh",
	function(g,x) return 1 - torch.pow(torch.tanh(x), 2.0) end,
}
gradfuns[torch.abs] = {
	"abs",
	function(g,x)
		if torch.isTensor(x) then
			return elemwise_mult(g,torch.sign(x))
		else
			sign = x>0 and 1 or x<0 and -1 or 0
			return elemwise_mult(g,sign)
		end
	end
}


-- First, override all the Torch functions
for ifn,fn_name in pairs(override) do
	local old = torch[fn_name]
	local new_fn = function(...)
		print("Running new " .. fn_name)
		return nodeapply(old, ...)
	end

	torch[fn_name] = new_fn
end

-- Now, override metatables for tensors
-- (this is because operations like A * 3 call into the metatable
-- for A. So if A is a DoubleTensor, Lua will look in getmetatable(torch.DoubleType)[__mul]
-- I THINK
local tensor_types = {
	'FloatTensor',
	'DoubleTensor'
}
for _,tensor_type in pairs(tensor_types) do
	local mt = torch.getmetatable('torch.' .. tensor_type)
	for ifn,fn_name in pairs(override) do
		local old = mt[fn_name]
		local new_fn = function(...)
			print("Running metatable " .. fn_name)
			return nodeapply(old, ...)
		end
		rawset(mt, fn_name, new_fn)
	end
end