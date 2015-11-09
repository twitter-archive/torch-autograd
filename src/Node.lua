local Value = require 'autograd.Value'
local Source = require 'autograd.Source'

Node = { }
Node.__index = Node

function Node.new(forwardFn, gradientFn, inputs)
	local v = { }
	setmetatable(v, Node)
	v:init(forwardFn, gradientFn, inputs)
	return v
end

function Node:init(forwardFn, gradientFn, inputs)
	self.forwardFn = forwardFn
	self.gradientFn = gradientFn
	self.inputs = { }
	for i = 1, #inputs do
		local input = inputs[i]
		if Value.isValue(input) then
			self.inputs[i] = input
		else
			self.inputs[i] = Value.from(input, Source.constant(input))
		end
	end
	self.gradients = { }
	self.outputs = { }
	self.outputTargets = { }
end

function Node:differentiable()
	if self.__differentiable == nil then
		for i = 1, #self.inputs do
			if self.inputs[i].source:differentiable() then
				self.__differentiable = true
				return true
			end
		end
		self.__differentiable = false
		return false
	else
		return self.__differentiable
	end
end

function Node:evaluateForward()
	local evalArgs = { }
	for i = 1, #self.inputs do
		local input = self.inputs[i]
		local source = input.source
		if source.type == Source.COMPUTED then
			local outputTargets = source.node.outputTargets[source.index]
			outputTargets[#outputTargets + 1] = {
				node = self,
				index = i
			}
		end
		evalArgs[i] = self.inputs[i]:flatten()
	end
	self.outputs = { }
	self.outputTargets = { }
	local outputs = {self.forwardFn.fn(unpack(evalArgs))}
	for i = 1, #outputs do
		self.outputs[i] = Value.from(outputs[i], Source.computed(self, i))
		self.outputTargets[i] = { }
	end
	return unpack(self.outputs)
end

function Node:evaluateBackward()
	for o = 1, #self.outputs do
		local output = self.outputs[o]
		for i = 1, #self.inputs do
			local input = self.inputs[i]
			local source = input.source
			if source:differentiable() then
				if self.gradients[o] == nil then
					if output.type == Value.TENSOR then
						-- TODO CORRECT TENSOR TYPE
						self.gradients[o] = Value.from(torch.FloatTensor(output:get():size()):zero(), Source.gradient(0, output:get():size()))
					elseif output.type == Value.NUMBER then
						self.gradients[o] = Value.from(0.0, Source.gradient(0))
					end
				end
				local gradUpdate = (self.gradientFn[i])(self.gradients[o], output, unpack(self.inputs))
				if gradUpdate then
					local sourceIndex = source.index or 1
					local gradSource = source.node or source
					if gradSource.gradients == nil then
						gradSource.gradients = { }
					end
					if gradSource.gradients[sourceIndex] == nil or gradSource.gradients[sourceIndex] == 0 then
						gradSource.gradients[sourceIndex] = gradUpdate
					else
						gradSource.gradients[sourceIndex] = gradSource.gradients[sourceIndex] + gradUpdate
					end
				end
			end
		end
	end
end

return Node