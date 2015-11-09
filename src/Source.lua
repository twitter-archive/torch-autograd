local Source = { }

Source.__index = Source

Source.COMPUTED = "computed"
Source.PARAM = "param"
Source.CONSTANT = "constant"
Source.TABLE = "table"
Source.GRADIENT = "gradient"

function Source.new(type)
	local v = { }
	setmetatable(v, Source)
	v:init(type)
	return v
end

function Source:init(type)
	self.type = type
end

function Source.computed(node, index)
	local s = Source.new(Source.COMPUTED)
	s.node = node
	s.index = index
	return s
end

function Source:symbolPath(rootSymbols)
	if self.type == Source.TABLE then
		if type(self.key) == 'number' then
			return self.parent:symbolPath(rootSymbols) .. "[" .. self.key .. "]"
		else
			return self.parent:symbolPath(rootSymbols) .. "." .. self.key
		end
	elseif self.type == Source.CONSTANT then
		if type(self.val) == "userdata" and self.val.totable then
			local tt = self.val:totable()
			return table.concat(tt, ", ")
		else
			return tostring(self.val)
		end
	elseif self.type == Source.GRADIENT then
		-- TODO TENSOR
		return tostring(self.val)
	else
		return rootSymbols[self]
	end
end

function Source:differentiable()
	if self.type == Source.TABLE then
		return self.parent:differentiable()
	elseif self.type == Source.COMPUTED then
		return self.node:differentiable()
	elseif self.type == Source.PARAM then
		return self.gradient
	end
end

function Source:getRoot()
	if self.type == Source.TABLE then
		return self.parent:getRoot()
	else
		return self
	end
end

function Source.param(name, gradient)
	local s = Source.new(Source.PARAM)
	s.name = name
	s.gradient = gradient
	return s
end

function Source.constant(value)
	local s = Source.new(Source.CONSTANT)
	s.val = value
	return s
end

function Source.table(parent, key)
	local s = Source.new(Source.TABLE)
	s.parent = parent
	s.key = key
	return s
end

function Source.gradient(val, dims)
	local s = Source.new(Source.GRADIENT)
	s.val = val
	s.dims = dims
	return s
end

return Source