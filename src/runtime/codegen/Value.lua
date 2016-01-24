local Source = require 'autograd.runtime.codegen.Source'

local Value = { }

Value.TABLE = "table"
Value.TENSOR = "tensor"
Value.NUMBER = "number"
Value.LONG_STORAGE = "long_storage"
Value.BOOLEAN = "boolean"

function Value.create(type, val, source)
	local v = {
		type = type,
		raw = val,
		source = source
	}
	setmetatable(v, Value)
	return v
end

function Value.from(v, source, skipWrapTables, mutationFlow)
	if v == nil then
		error("nil parameter value")
	elseif Value.isValue(v) then
		if mutationFlow ~= nil then
			return mutationFlow:remap(v)
		end
		return v
	elseif type(v) == "table" then
		local vcopy = { }
		for k,v in pairs(v) do
			vcopy[k] = Value.from(v, Source.table(source, k), skipWrapTables, mutationFlow)
		end
		if skipWrapTables then
			return vcopy
		else
			return Value.create(Value.TABLE, vcopy, source)
		end
	elseif torch.isTensor(v) then
		return Value.create(Value.TENSOR, v, source)
	elseif type(v) == "number" then
		return Value.create(Value.NUMBER, v, source)
	elseif type(v) == "boolean" then
		return Value.create(Value.BOOLEAN, v, source)
	elseif v.totable then
		return Value.create(Value.LONG_STORAGE, v, source)
	else
		error("unknown type " .. type(v))
	end
end

function Value.isValue(v)
	return getmetatable(v) == Value
end

function Value.len(v)
	if Value.isValue(v) then
		return #v.raw
	else
		return #v
	end
end

function Value:get()
	return self.raw
end

function Value.__internal_set(s, k, v)
	s[k] = v
	return s
end

function Value.__internal_get(s, k)
	return s[k]
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
		if type(i) ~= "string" then
			return Value.__internal_get(self, i)
		else
			if raw[i] ~= nil then
				return raw[i]
			end
		end
	end
	return rawget(Value, i)
end

function Value:__newindex(k, v)
	local rtype = rawget(self, "type")
	if rtype == Value.TABLE then
		local raw = rawget(self, "raw")
		return rawset(raw, k, v)
	elseif rtype == Value.TENSOR then
		local raw = rawget(self, "raw")
		if type(k) ~= "string" then
			return Value.__internal_set(self, k, v)
		end
	end
	return rawset(self, k, v)
end

function Value:__len()
	return #self.raw
end

function Value.flatten(v)
	if not Value.isValue(v) then
		if type(v) == "table" then
			local rawTable = { }
			for k,v in pairs(v) do
				rawTable[k] = Value.flatten(v)
			end
			return rawTable
		else
			return v
		end
	elseif v.type == Value.TABLE then
		return Value.flatten(v.raw)
	else
		return v.raw
	end
end

function Value.flattenGrads(v, intermediateGrads)
	if not Value.isValue(v) then
		if type(v) == "table" then
			local rawTable = { }
			for k,v in pairs(v) do
				rawTable[k] = Value.flattenGrads(v, intermediateGrads)
			end
			return rawTable
		end
	elseif v.type == Value.TABLE then
		return Value.flattenGrads(v.raw, intermediateGrads)
	else
		local grad = intermediateGrads[v.source]
		if grad ~= nil then
			return intermediateGrads[v.source]:flatten()
		end
	end
end

function Value.collectGrads(v, intermediateGrads)
	if not Value.isValue(v) then
		if type(v) == "table" then
			local rawTable = { }
			for k,v in pairs(v) do
				rawTable[k] = Value.collectGrads(v, intermediateGrads)
			end
			return rawTable
		end
	elseif v.type == Value.TABLE then
		return Value.collectGrads(v.raw, intermediateGrads)
	else
		return intermediateGrads[v.source]
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