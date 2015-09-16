-- TODO
-- Disallow overwriting anything
-- 

torch = torch or require('torch')
class = require 'class'
require 'util'
debug = require 'debug'
-- local STP = require "StackTracePlus"
-- debug.traceback = STP.stacktrace

function to_scalar(x)
	return torch.sum(torch.sin(x))
end

local op = {
	add = function(a,b) return a+b end,
	sub = function(a,b) return a-b end,
	mul = function(a,b) return a*b end,
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
	return nodeapply(op.mul, l, r)
end

function Node.__div(l,r)
	return nodeapply(op.div, l, r)
end
function Node.__pow(l,r)
	return nodeapply(op.pow, l, r)
end
function Node.__unm(l)
	return nodeapply(op.mul, -1, l)
end

local number_mt = {
	__add = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			return nodeapply(op.add, b, a)
		else
			return nodeapply(op.add, a, b)
		end
	end,
	__sub = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			return nodeapply(op.add, -b, a)
		else
			return nodeapply(op.add, a, -b) -- TODO subtraction
		end
	end,
	__mul = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			return nodeapply(op.mul, b, a)
		else
			return nodeapply(op.mul, a, b)
		end
	end,
	__div = function(a,b)
		if torch.type(a) == "number" and torch.isTensor(b) then
			-- THIS IS INSANE
			c = torch.ones(b:size())
			return node.apply(op.mul, torch.cdiv(c,b), a)
		else
			return node.apply(op.div, a, b)
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
    -- print("")
    if count(parents) > 0 then
        local vals = map(getval,arg)
        local value = nodeapply(fun,unpack(map(getval,arg)))
        return Node(value, fun, arg, parents[1].tape)
    else
        return fun(unpack(map(getval,arg)))
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

		-- print(map(function(t)
		-- 		if t.fun then
		-- 			return gradfuns[t.fun][1]
		-- 		else
		-- 			return nil
		-- 		end
		-- 	end,
		-- 	ans.tape))

		for i=#ans.tape,1,-1 do
			node = ans.tape[i]
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

local function elemwise_mul(a,b)
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

local gradMt = {}
gradMt.__index = function(table, key)
	print("")
	print(debug.getinfo(key))
	error("No adjoint found for function " .. tostring(key) .. ". Debug info above.")
end
setmetatable(gradfuns, gradMt)

gradfuns[op.add] = {
	"add",
	function(g, x, y) return g end,
	function(g, x, y) return g end,
}
gradfuns[op.mul] = {
	"mult/dot",
	function(g, A, B)
		if torch.isTensor(A) and torch.isTensor(B) then
			if B:nDimension() == 2 then
				return g*B:t()
			elseif A:nDimension() == 2 then
				return torch.ger(g, B) -- outer product
			else
				return g*B -- elemwise_mul required? what about 3D?
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
	function(g, x, y) return elemwise_mul(-g,elemwise_div(x,torch.pow(y,2))) end,
}
gradfuns[op.sub] = {
	"sub",
	function(g, x, y) return g end,
	function(g, x, y) return -g end,
}
gradfuns[op.pow] = {
	"pow",
	function(g, x, y) return elemwise_mul(elemwise_mul(g,y),torch.pow(x, (y-1))) end,
}
-- gradfuns.__mul = gradfuns[op.mul]
-- gradfuns.__add = gradfuns[op.add]
-- gradfuns.__div = gradfuns[op.div]
-- gradfuns.__pow = gradfuns[op.pow]
-- gradfuns.__sub = gradfuns[op.sub]
gradfuns[torch.add] = {
	"torch.add",
	function(g, x, y) return g end,
	function(g, x, y) return g end,
}
gradfuns[torch.cmul] = {
	"torch.cmul",
	function(g, x, y) return elemwise_mul(y,g) end,
	function(g, x, y) return elemwise_mul(x,g) end,	
}
gradfuns[torch.mul] = {
	"torch.mul",
	function(g, x, y) return elemwise_mul(y,g) end,
	function(g, x, y) return elemwise_mul(x,g) end,	
}
gradfuns[torch.pow] = {
	"torch.pow",
	function(g, x, y) return elemwise_mul(elemwise_mul(g,y),torch.pow(x,y-1)) end
}
gradfuns[torch.exp] = {
	"exp",
	function(g,x) return elemwise_mul(torch.exp(x), g) end,
}
gradfuns[torch.tanh] = {
	"tanh",
	function(g,x) return 1 - torch.pow(torch.tanh(x), 2.0) end,
}
gradfuns[torch.abs] = {
	"abs",
	function(g,x)
		if torch.isTensor(x) then
			return elemwise_mul(g,torch.sign(x))
		else
			sign = x>0 and 1 or x<0 and -1 or 0
			return elemwise_mul(g,sign)
		end
	end
}
gradfuns[torch.sum] = {
	"sum",
	function(g,x,axis)
		if axis then
			local sizes = x:size():fill(1)
			sizes[axis] = x:size(axis)
			return torch.repeatTensor(g, sizes)
		else
			return x:clone():fill(g)
		end
		return g
	end
}
gradfuns[torch.sqrt] = {
	"sqrt",
	function(g,x) return elemwise_mul(elemwise_mul(g,0.5), torch.pow(x,-0.5)) end
}
gradfuns[torch.sin] = {
	"sin",
	function(g,x) return elemwise_mul(g, torch.cos(x)) end
}
gradfuns[torch.cos] = {
	"cos",
	function(g,x) return elemwise_mul(g, -torch.sin(x)) end
}
gradfuns[torch.tan] = {
	"tan",
	function(g,x) return elemwise_div(g, torch.pow(torch.cos(x), 2.0)) end
}

local override = { 
	"__add", "add", 
	"__mul", "mul", "cmul",
	"pow", "__pow",
	"exp", 'tanh',
	'abs', 'sum'
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


-- Now override class methods and metamethods on tensors
local tensor_types = {
	'FloatTensor',
	'DoubleTensor'
}

-- Override metamethods like __mul and __add
elem_op_override = {
	__mul = op.mul,
	__sub = op.sub,
	__div = op.div,
	__add = op.add,
}
for _,tensor_type in pairs(tensor_types) do
	local mt = torch.getmetatable('torch.' .. tensor_type)
	for src,dest in pairs(elem_op_override) do
		gradfuns[mt[src]] = gradfuns[dest]
	end
end

-- Make sure that all class functions
-- hook into the autodiff engine
-- (so that we capture evaluations of torch.sum() and also myTensor:sum())
for _,tensor_type in pairs(tensor_types) do
	local mt = torch.getmetatable('torch.' .. tensor_type)
	for ifn,fn_name in pairs(override) do
		local old = mt[fn_name]
		local new_fn = function(...)
			print("Running metamethod " .. fn_name)
			return nodeapply(old, ...)
		end
		rawset(mt, fn_name, new_fn)
	end
end