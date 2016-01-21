local MutationFlow = { }

MutationFlow.__index = MutationFlow

function MutationFlow.new()
	local v = {
		history = { },
		map = { },
		reverseMap = { },
	}
	setmetatable(v, MutationFlow)
	return v
end

function MutationFlow:alias(from, to)
	self.history[#self.history + 1] = {
	   from = from,
	   to = to,
	}
	local reverse = self.reverseMap[from]
	if reverse ~= nil then
	   self.map[reverse] = to
	end
	self.map[from] = to
	self.reverseMap[to] = from
end

function MutationFlow:remap(a)
	local alias = self.map[a]
	if alias ~= nil then
	   return alias
	else
	   return a
	end
end

function MutationFlow:clear()
	self.history = { }
	self.map = { }
end

return MutationFlow