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

function Source:symbolPath(rootSymbols)
	if self.type == Source.TABLE then
		if type(self.key) == 'number' then
			return self.parent:symbolPath(rootSymbols) .. "[" .. self.key .. "]"
		else
			return self.parent:symbolPath(rootSymbols) .. "." .. self.key
		end
	elseif self.type == Source.CONSTANT then
		if type(self.val) == "userdata" and self.val.totable ~= nil then
			if torch.isTensor(self.val) then
				if rootSymbols[self] then
					return rootSymbols[self]
				else
					local tt = self.val:totable()
					return self.val:type() .. "({" .. table.concat(tt, ", ") .. "})"
				end
			else
				local tt = self.val:totable()
				return table.concat(tt, ", ")
			end
		elseif type(self.val) == "table" then
			local Value = require 'autograd.runtime.codegen.Value'
			local elements = { }
			for k, v in pairs(self.val) do
				if Value.isValue(v) then
					elements[#elements + 1] = v.source:symbolPath(rootSymbols)
				else
					elements[#elements + 1] = tostring(v)
				end
			end
			return "{" .. table.concat(elements, ", ") .. "}"
		else
			if self.val == math.huge then
				return "math.huge"
			end
			return tostring(self.val)
		end
	elseif self.type == Source.GRADIENT then
		-- TODO TENSOR
		if self.tensorType ~= nil then
			return "(" .. self.tensorType .. "(" .. table.concat(self.dims:totable(), ", ") .. "):fill(" .. self.val .. ")" .. ")"
		else
			return tostring(self.val)
		end
	else
		if rootSymbols[self] == nil then
			error("unknown symbol for node")
		end
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
	elseif self.type == Source.CONSTANT then
		if type(self.val) == "table" then
			local Value = require 'autograd.runtime.codegen.Value'
			for k, v in pairs(self.val) do
				if Value.isValue(v) then
					if v.source:differentiable() then
						return true
					end
				end
			end
		end
	end
	return false
end

function Source:getRoot()
	if self.type == Source.TABLE then
		return self.parent:getRoot()
	else
		return self
	end
end

function Source:changeRoot(newRoot)
	if self.type == Source.TABLE then
		if self.parent.type ~= Source.TABLE then
			self.parent = newRoot
		else
			return self.parent:changeRoot(newRoot)
		end
	else
		return newRoot
	end
end

function Source:getParentsArray(arr)
	arr = arr or { }
	if self.type == Source.TABLE then
		self.parent:getParentsArray(arr)
	end
	arr[#arr + 1] = self
	return arr
end

function Source:changeNodeTargetIndex(target, currentIdx, newIdx)
	if self.type == Source.COMPUTED then
		self.node:changeTargetIndex(self.index, target, currentIdx, newIdx)
	end
end

function Source.computed(node, index)
	local s = Source.new(Source.COMPUTED)
	s.node = node
	s.index = index
	return s
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

function Source.gradient(val, tensorType, dims)
	local s = Source.new(Source.GRADIENT)
	s.val = val
	s.tensorType = tensorType
	s.dims = dims
	return s
end

return Source