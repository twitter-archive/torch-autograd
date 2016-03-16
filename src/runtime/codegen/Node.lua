local Value = require 'autograd.runtime.codegen.Value'
local Source = require 'autograd.runtime.codegen.Source'
local util = require 'autograd.util'

Node = { }
Node.__index = Node

function Node.new(forwardFn, gradientFn, inputs, mutationFlow)
	local v = { }
	setmetatable(v, Node)
	v:init(forwardFn, gradientFn, inputs, mutationFlow)
	return v
end

function Node:init(forwardFn, gradientFn, inputs, mutationFlow)
	self.forwardFn = forwardFn
	self.gradientFn = gradientFn
	self.inputs = { }
	for i = 1, #inputs do
		local input = inputs[i]
		if not Value.isValue(input) then
			if torch.isTensor(input) and torch.nDimension(input) > 1 then
				error("constant tensor with more than one dimension. is this an upvalue that should be a function argument?")
			end
		end
		self.inputs[i] = Value.from(input, Source.constant(input), false, mutationFlow)
	end
	self.outputs = { }
	self.outputTargets = { }
end

function Node:differentiable(differentiableMap)
	local outputSource = self.outputs[1].source
	local isDiff = differentiableMap[outputSource]
	if isDiff == nil then
		if self.gradientFn or self.forwardFn.differentiable then
			for i = 1, #self.inputs do
				if self.inputs[i].source:differentiable(differentiableMap) then
					differentiableMap[outputSource] = true
					return true
				end
			end
		end
		differentiableMap[outputSource] = false
		return false
	else
		return isDiff
	end
end

function Node:evaluateForward(mutationFlow)
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
	if self.forwardFn.name == "Value.__internal_set" then
		-- This was an mutation in the form of x[k] = v
		-- The output of the assignment is simply the input param x wrapped in a new Value pointing to this node, x'
		-- All future user references to x will be remapped to x', to preserve the order of operations in the graph.
		local valueAlias = Value.from(outputs[1], Source.computed(self, 1))
		mutationFlow:alias(self.inputs[1], valueAlias)
		self.outputs[1] = valueAlias
		self.outputTargets[1] = { }
	else
		for i = 1, #outputs do
			self.outputs[i] = Value.from(outputs[i], Source.computed(self, i))
			self.outputTargets[i] = { }
		end
	end
	return table.unpack(self.outputs)
end

function Node:evaluateBackward(mutationFlow, intermediateGrads, differentiableMap)
	-- Only eval one gradient for now?
	local numGrads = 1 --#self.outputs
	for o = 1, numGrads do
		local output = self.outputs[o]
		for i = #self.inputs, 1, -1 do
			local input = self.inputs[i]
			local source = input.source
			if source:differentiable(differentiableMap) then
				if self.gradientFn ~= nil and self.gradientFn[i] ~= nil then
					local outputGradient = intermediateGrads[output.source]
					if outputGradient == nil then
						if output.type == Value.TENSOR then
						   outputGradient = Value.from(util.zerosLike(output), Source.constant(0, torch.type(output), torch.size(output)))
						elseif output.type == Value.NUMBER then
						   outputGradient = Value.from(0.0, Source.constant(0))
						end
						intermediateGrads[output.source] = outputGradient
					end
					if input.type == Value.TABLE then
						local gradUpdates = (self.gradientFn[i])(outputGradient, output, table.unpack(self.inputs))
						if gradUpdates then
							for k, v in pairs(input:get()) do
								local gradUpdate = mutationFlow:remap(gradUpdates[k])
								if gradUpdate ~= nil then
									local subArg = v
									local source = subArg.source
									local sourceGradient = intermediateGrads[source]
									if sourceGradient == nil or sourceGradient == 0 then
										intermediateGrads[source] = gradUpdate
									else
										intermediateGrads[source] = sourceGradient + gradUpdate
									end
								end
							end
						end
					else
						local gradUpdate = (self.gradientFn[i])(outputGradient, output, table.unpack(self.inputs))
						if gradUpdate then
							gradUpdate = mutationFlow:remap(gradUpdate)
							local sourceGradient = intermediateGrads[source]
							if sourceGradient == nil or sourceGradient == 0 then
								intermediateGrads[source] = gradUpdate
							else
								intermediateGrads[source] = sourceGradient + gradUpdate
							end
						end
					end
				elseif self.forwardFn.differentiable then
					error("missing gradient for argument " .. tostring(i) .. " in function " .. self.forwardFn.name)
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