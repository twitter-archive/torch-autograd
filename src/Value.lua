local Source = require 'autograd.Source'

local Value = { }

Value.TABLE = "table"
Value.TENSOR = "tensor"
Value.NUMBER = "number"
Value.LONG_STORAGE = "long_storage"

function Value.create(type, val, source)
	local v = { }
	setmetatable(v, Value)
	v:init(type, val, source)
	return v
end

function Value:init(type, val, source)
	self.type = type
	self.raw = val
	self.source = source
end

function Value.from(v, source)
	if Value.isValue(v) then
		error("already a value type")
	elseif type(v) == "table" then
		local vcopy = { }
		for k,v in pairs(v) do
			vcopy[k] = Value.from(v, Source.table(source, k))
		end
		return Value.create(Value.TABLE, vcopy, source)
	elseif torch.isTensor(v) then
		return Value.create(Value.TENSOR, v, source)
	elseif type(v) == "number" then
		return Value.create(Value.NUMBER, v, source)
	elseif v.totable then
		return Value.create(Value.LONG_STORAGE, v, source)
	else
		error("unknown type " .. type(v))
	end
end

function Value.isValue(v)
	return getmetatable(v) == Value
end

function Value:get()
	return self.raw
end

function Value:__index(i)
	local rtype = rawget(self, "type")
	if rtype == Value.TABLE then
		local raw = rawget(self, "raw")
		if raw[i] ~= nil then
			return raw[i]
		end
	elseif rtype == Value.TENSOR then
		local raw = rawget(self, "raw")
		if raw[i] ~= nil then
			return raw[i]
		end
	end
	return rawget(Value, i)
end

function Value:flatten()
	if self.type == Value.TABLE then
		local rawTable = { }
		for k,v in pairs(self.raw) do
			rawTable[k] = v:flatten()
		end
		return rawTable
	else
		return self.raw
	end
end

-- These exist only to be overloaded and called with flattened tensor or number arguments

function Value.__add(a, b)
	return a + b
end

function Value.__sub(a, b)
	return a - b
end

function Value.__mul(a, b)
	return a * b
end

function Value.__div(a, b)
	return a / b
end

function Value.__pow(a, b)
	return a ^ b
end

function Value.__unm(a)
	return -a
end

return Value