local Value = require 'autograd.runtime.codegen.Value'
local Source = require 'autograd.runtime.codegen.Source'

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
			if torch.isTensor(input) then
				if torch.nDimension(input) > 1 then
					error("constant tensor with more than one dimension. is this an upvalue that should be a function argument?")
				end
			end
			self.inputs[i] = Value.from(input, Source.constant(input))
		end
	end
	self.gradients = { }
	self.outputs = { }
	self.outputTargets = { }
end

function Node:differentiable()
	if self.__differentiable == nil then
		if self.gradientFn or self.forwardFn.differentiable then
			for i = 1, #self.inputs do
				if self.inputs[i].source:differentiable() then
					self.__differentiable = true
					return true
				end
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
			source.node:linkOutputNode(source.index, self, i)
		elseif source.type == Source.CONSTANT and input.type == Value.TABLE then
			-- Constant table assembled by the user.
			for k, v in pairs(input:get()) do
				if Value.isValue(v) then
					if v.source.type == Source.COMPUTED then
						v.source.node:linkOutputNode(v.source.index, self, i)
					end
				end
			end
		end
		evalArgs[i] = self.inputs[i]:flatten()
	end
	self.outputs = { }
	self.outputTargets = { }
	local outputs = {self.forwardFn.fn(table.unpack(evalArgs))}
	for i = 1, #outputs do
		self.outputs[i] = Value.from(outputs[i], Source.computed(self, i))
		self.outputTargets[i] = { }
	end
	return table.unpack(self.outputs)
end

function Node:evaluateBackward()
	-- Only eval one gradient for now?
	local numGrads = 1 --#self.outputs
	for o = 1, numGrads do
		local output = self.outputs[o]
		for i = 1, #self.inputs do
			local input = self.inputs[i]
			local source = input.source
			if source:differentiable() then
				if self.gradientFn ~= nil and self.gradientFn[i] ~= nil then
					if self.gradients[o] == nil then
						if output.type == Value.TENSOR then
							local tens = output:get()
							self.gradients[o] = Value.from(torch[tens:type():gsub("torch%.", "")](tens:size()):zero(), Source.gradient(0, tens:type(), tens:size()))
						elseif output.type == Value.NUMBER then
							self.gradients[o] = Value.from(0.0, Source.gradient(0))
						end
					end
					if input.type == Value.TABLE then
						local gradUpdates = (self.gradientFn[i])(self.gradients[o], output, table.unpack(self.inputs))
						if gradUpdates then
							for k, v in pairs(input:get()) do
								local gradUpdate = gradUpdates[k]
								local subArg = v
								local source = subArg.source
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
					else
						local gradUpdate = (self.gradientFn[i])(self.gradients[o], output, table.unpack(self.inputs))
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
				elseif self.forwardFn.differentiable then
					error("missing gradient for function " .. self.forwardFn.name)
				end
			end
		end
	end
end

local function removeFromTargetsArray(arr, node)
   for i = #arr, 1, -1 do
      if arr[i].node == node then
         table.remove(arr, i)
      end
   end
end

function Node:unlinkInputs()
	for i = 1, #self.inputs do
		if self.inputs[i].source.type == Source.COMPUTED then
			self.inputs[i].source.node:unlinkOutputNode(self)
		end
	end
	self.inputs = { }
end

function Node:replaceInput(replaceInput, withInput)
	for i = 1, #self.inputs do
		local input = self.inputs[i]
		if input == replaceInput then
			if replaceInput.source.type == Source.COMPUTED then
				replaceInput.source.node:unlinkOutputNode(self)
			end
			if withInput.source.type == Source.COMPUTED then
				local inputIndex = withInput.source.node:outputParamIndex(withInput)
				withInput.source.node:linkOutputNode(inputIndex, self, i)
			end
			self.inputs[i] = withInput
		end
	end
end

function Node:linkOutputNode(srcIndex, node, dstIndex)
	local outputTargets = self.outputTargets[srcIndex]
	outputTargets[#outputTargets + 1] = {
		node = node,
		index = dstIndex
	}
end

function Node:unlinkOutputNode(node)
	for k = 1, #self.outputTargets do
		removeFromTargetsArray(self.outputTargets[k], node)
	end
end

function Node:outputParamIndex(outputValue)
	for k = 1, #self.outputs do
		if self.outputs[k] == outputValue then
			return k
		end
	end
	return 0
end

function Node:changeTargetIndex(param, target, currentIdx, newIdx)
	for i = 1, #self.outputTargets[param] do
		local ot = self.outputTargets[param][i]
		if ot.node == self and ot.index == currentIdx then
			out.index = newIdx
		end
	end
end

return Node