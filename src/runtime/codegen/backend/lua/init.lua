local Value = require 'autograd.runtime.codegen.Value'
local Source = require 'autograd.runtime.codegen.Source'
local StringBuilder = require 'autograd.StringBuilder'
local Debugger = require 'autograd.runtime.codegen.Debugger'
local util = require 'autograd.util'

local reusableFunctionsMap = {
   ["torch.tanh"] = true,
   ["torch.cmul"] = true,
   ["torch.cdiv"] = true,
   ["torch.exp"]  = true,
   ["torch.pow"]  = true,
   ["torch.add"]  = true,
   ["torch.mul"]  = true,
   ["torch.div"] = true,
   ["torch.neg"]  = true,
   ["torch.ger"]  = true,
   ["torch.mm"]   = true,
   ["torch.mv"]   = true,
   ["torch.cosh"] = true,
   ["torch.cat"] = true,
   ["torch.log"] = true,
   ["torch.repeatTensor"] = true,
   ["util.sigmoidInPlace"] = true,
   ["util.narrowSliceCopyInPlace"] = true,
   ["util.selectSliceCopyInPlace"] = true,
   ["util.fillSameSizeAsInPlace"] = true,
   ["util.fillSameSizeAsInPlace"] = true,
   ["util.zerosLikeInPlace"] = true,
   ["util.setNotEqualInPlace"] = true,
   ["util.indexAddInPlace"] = true,
   ["util.newTensorLikeInPlace"] = true,
   ["util.fillInPlace"] = true,
   ["util.cloneInPlace"] = true,
   ["util.newInPlace"] = true,
   ["util.typeAsInPlace"] = true,
}

local reusableFunctionTransforms = {
   ["util.narrowSliceCopy"] = "util.narrowSliceCopyInPlace",
   ["util.selectSliceCopy"] = "util.selectSliceCopyInPlace",
   ["util.fillSameSizeAs"] = "util.fillSameSizeAsInPlace",
   ["util.zerosLike"] = "util.zerosLikeInPlace",
   ["util.setNotEqual"] = "util.setNotEqualInPlace",
   ["util.indexAdd"] = "util.indexAddInPlace",
   ["util.sigmoid"] = "util.sigmoidInPlace",
   ["util.newTensorLike"] = "util.newTensorLikeInPlace",
   ["util.fill"] = "util.fillInPlace",
   ["torch.clone"] = "util.cloneInPlace",
   ["torch.DoubleTensor.new"] = "util.newInPlace",
   ["torch.FloatTensor.new"] = "util.newInPlace",
   ["torch.CudaTensor.new"] = "util.newInPlace",
   ["torch.typeAs"] = "util.typeAsInPlace",
}

local function canReuseOutput(node)
   return reusableFunctionsMap[node.forwardFn.name] ~= nil and #node.outputs == 1 and node.outputs[1].type == Value.TENSOR
end

local function canInline(node, state)
   return #node.outputs == 1 and #node.outputTargets[1] == 1 and state.hazardNodes[node] == nil and state.debugger == nil and state.inline
end

local function writeLiteralTable(wtable, out, symbols, depth)
   depth = depth or 1
   out.write("{", "\n")
   local keys = { }
   local numeric = true
   for k, v in pairs(wtable) do
      if type(k) ~= 'number' then
         numeric = false
      end
      keys[#keys + 1] = k
   end
   local si = #keys
   local ei = 1
   local di = -1
   if numeric then
      si = 1
      ei = #keys
      di = 1
   end
   for i = si, ei, di do
      local k = keys[i]
      local v = wtable[k]
      out.write(string.rep(" ", depth * 4))
      if type(k) == 'number' or tostring(tonumber(k)) == k then
         out.write("[", tostring(k), "]")
      else
         out.write(tostring(k))
      end
      out.write(" = ")
      if Value.isValue(v) then
         out.write(v.source:symbolPath(symbols))
      elseif type(v) == 'table' then
         writeLiteralTable(v, out, symbols, depth + 1)
      else
         out.write(tostring(v))
      end
      out.write(",\n")
   end
   out.write(string.rep(" ", (depth-1) * 4), "}")
end

local writeExpr

local function buildInputExpr(state, input)
   if input.source.type == Source.CONSTANT and type(input.source.val) == "table" then
      -- Literal table.
      -- TODO: Only supports arrays.
      local elements = { }
      for k, v in pairs(input.source.val) do
         if Value.isValue(v) then
            elements[#elements + 1] = buildInputExpr(state, v)
         else
            elements[#elements + 1] = tostring(v)
         end
      end
      return "{" .. table.concat(elements, ", ") .. "}"
   elseif input.source.type == Source.COMPUTED and canInline(input.source.node, state) then
      local subExpr = writeExpr(state, input.source.node)
      return "(" .. subExpr .. ")"
   else
      local symbol = input.source:symbolPath(state.symbols)
      return symbol
   end
end

writeExpr = function(state, node)
   local out = StringBuilder()
   local inputSymbols = { }
   for k = 1, #node.inputs do
      local input = node.inputs[k]
      inputSymbols[k] = buildInputExpr(state, input)
   end
   if node.forwardFn.operator ~= nil then
      local op = node.forwardFn.operator
      if op == "unm" then
         out.write("-", inputSymbols[1])
      elseif op == "index" then
         out.write(inputSymbols[1])
         out.write("[")
         out.write(inputSymbols[2])
         out.write("]")
      elseif op == "newindex" then
         out.write(inputSymbols[1])
         out.write("[")
         out.write(inputSymbols[2])
         out.write("]")
         out.write(" = ")
         out.write(inputSymbols[3])
      else
         out.write(inputSymbols[1])
         out.write(" ")
         if op == "add" then
            out.write("+")
         elseif op == "sub" then
            out.write("-")
         elseif op == "mul" then
            out.write("*")
         elseif op == "div" then
            out.write("/")
         end
         out.write(" ")
         out.write(inputSymbols[2])
      end
   elseif node.forwardFn.object ~= nil then
      out.write(state.objects[node.forwardFn.object].name, ".", node.forwardFn.method, "(", table.concat(inputSymbols, ", "), ")")
   else
      local fnName = node.forwardFn.name
      if canReuseOutput(node) then
         table.insert(inputSymbols, 1, node.outputs[1].source:symbolPath(state.symbols))
      end
      out.write(state.functionRemap[fnName], "(", table.concat(inputSymbols, ", "), ")")
   end
   return out.finish()
end

local function tensorSig(t)
   return t:type() .. table.concat(t:size():totable(), ",")
end

local function storageSig(t)
   return torch.type(t) .. table.concat(t:totable(), ",")
end

local function letterForType(val)
   if val.type == Value.TENSOR then
      return "t"
   elseif val.type == Value.NUMBER then
      return "n"
   else
      return "r"
   end
end

local function searchMatchingTensorLargest(tensor, sortedTensors, locals)
   local ttype = tensor:type()
   for i = #sortedTensors, 1, -1 do
      local idx = sortedTensors[i]
      local lt = locals[idx]
      if lt:type() == ttype then
         return i
      end
   end
   return 0
end

local function findParamSource(val)
   if Value.isValue(val) then
      local rootSource = val.source:getRoot()
      if rootSource.type == Source.PARAM then
         return rootSource
      end
   elseif type(val) == "table" then
      for k, v in pairs(val) do
         local paramSource = findParamSource(v)
         if paramSource ~= nil then
            return paramSource
         end
      end
   end
end

local function collectParams(val, params, seen, depth)
   params = params or { }
   for k, v in pairs(val) do
      local paramSource = findParamSource(v)
      params[k] = paramSource or Source.param(k)
   end
   return params
end

local function flattenAnswer(val)
   if Value.isValue(val) then
      return val:flatten()
   elseif type(val) == "table" then
      local ft = { }
      for k, v in pairs(val) do
         ft[k] = flattenAnswer(v)
      end
      return ft
   else
      return val
   end
end

local function mapReusableTensorNodeSymbol(node, symbols, tensorPool, availableTensorMap, remainingOutputs, availableCount, index)
   local output = node.outputs[1]
   local tensor = output:get()
   local sig = tensorSig(tensor)
   local matchingList = availableTensorMap[sig]
   local tensorIdx = nil
   if matchingList ~= nil and #matchingList > 0 then
      -- Map to tensor pool.
      tensorIdx = table.remove(matchingList, #matchingList)
      availableCount = availableCount - 1
   else
      if availableCount > 0 and index ~= nil then
         -- There are tensors remaining, so keep track for possible later inexact allocation.
         remainingOutputs[#remainingOutputs + 1] = index
      else
         -- No available tensors, so just go ahead and allocate a slot for this one now.
         tensorIdx = #tensorPool + 1
         tensorPool[tensorIdx] = tensor
      end
   end
   if tensorIdx ~= nil then
      symbols[output.source] = "rlocals[" .. tensorIdx .. "]"
   end
   return availableCount
end

local function createSymbolTable(graph, execOrder, aliases, params, tensorPool, tensorLocals, opt)
   -- Assign symbols to params, inputs, outputs.
   local symbols = { }
   local undefined = { }
   local constants = { }

   for i = 1, #params do
      symbols[params[i]] = "p" .. i
   end

   local constantStorageMap = { }
   local constantTensorMap = { }

   local tensorPoolViews = { }

   local availableTensorMap = { }
   local availableCount = 0
   local tensorSigs = { }
   for i = #tensorPool, 1, -1 do
      local tensor = tensorPool[i]
      local sig = tensorSig(tensor)
      local list = availableTensorMap[sig]
      if list == nil then
         list = { }
         availableTensorMap[sig] = list
      end
      list[#list + 1] = i
      availableCount = availableCount + 1
   end

   local remainingOutputs = { }

   local localCount = 0

   local skip = { }

   -- Guarantee a stable mapping for gradient output tensors.
   if opt.stableGradients then
      local grads = graph.grads
      local gradsByParamPath = { }
      for i = 1, #grads do
         local node = grads[i].grad.source:getRoot().node
         if canReuseOutput(node) then
            local paramPath = grads[i].param.source:symbolPath(symbols)
            gradsByParamPath[paramPath] = node
         end
      end

      local flatParamGrads = util.sortedFlatten(gradsByParamPath, { }, true)
      for i = 1, #flatParamGrads do
         local gradNode = flatParamGrads[i]
         availableCount = mapReusableTensorNodeSymbol(gradNode, symbols, tensorPool, availableTensorMap, remainingOutputs, availableCount)
         skip[gradNode] = true
      end
   end

   -- Exact matches first.
   for i = 1, #execOrder do
      local node = execOrder[i]
      if aliases[node] ~= nil or skip[node] ~= nil then
      elseif #node.outputs == 1 then
         local output = node.outputs[1]
         if node.outputs[1].type == Value.TENSOR and canReuseOutput(node) then
               availableCount = mapReusableTensorNodeSymbol(node, symbols, tensorPool, availableTensorMap, remainingOutputs, availableCount, i)
         else
            -- Non-reusable local.
            localCount = localCount + 1
            tensorLocals[localCount] = 0
            symbols[output.source] = "locals[" .. localCount .. "]"
         end
         -- else
         --    -- One output, not a tensor.
         --    undefined[output.source] = true
         --    symbols[output.source] = letterForType(node.outputs[1]) .. i
         -- end
      else
         -- More than one output.
         -- TODO, currently uncached.
         for k = 1, #node.outputs do
            local output = node.outputs[k]
            undefined[output.source] = true
            symbols[node.outputs[k].source] = letterForType(node.outputs[k]) .. i .. "_" .. k
         end
      end

      -- Find constant inputs.
      for k = 1, #node.inputs do
         local input = node.inputs[k]
         local source = input.source:getRoot()
         if source.type == Source.CONSTANT and symbols[source] == nil then
            if torch.isTensor(source.val) then
               local index = constantTensorMap[source.val]
               if index == nil then
                  index = #constants + 1
                  constantTensorMap[source.val] = index
                  constants[index] = source
               end
               symbols[source] = "c" .. index
            elseif torch.isStorage(source.val) then
               local sig = storageSig(source.val)
               if constantStorageMap[sig] then
                  symbols[source] = "c" .. constantStorageMap[sig]
               else
                  index = #constants + 1
                  constants[index] = source
                  constantStorageMap[sig] = index
                  symbols[source] = "c" .. index
               end
            end
         end
      end
   end

   -- Did we fail to find a spot for any tensors? Try an inexact mapping that requires a view.

   local availableTensors = { }

   if availableCount > 0 then
      -- Only bother sorting the two lists by size if we're actually going to use them.
      local availableTensorSizes = { }
      for k, v in pairs(availableTensorMap) do
         for i = 1, #v do
            local idx = v[i]
            availableTensors[#availableTensors + 1] = idx
            availableTensorSizes[idx] = tensorPool[idx]:storage():size()
         end
      end

      function sortLocalSize(a, b)
        return availableTensorSizes[a] < availableTensorSizes[b]
      end

      table.sort(availableTensors, sortLocalSize)

      local remainingOutputSizes = { }
      for i = 1, #remainingOutputs do
         local idx = remainingOutputs[i]
         local output = execOrder[idx].outputs[1]
         remainingOutputSizes[idx] = output:get():storage():size()
      end

      function sortTensorSize(a, b)
        return remainingOutputSizes[a] < remainingOutputSizes[b]
      end

      table.sort(remainingOutputs, sortTensorSize)
   end

   for i = #remainingOutputs, 1, -1 do
      local output = execOrder[remainingOutputs[i]].outputs[1]
      local outputTensor = output:get()
      local matchingIndex = searchMatchingTensorLargest(outputTensor, availableTensors, tensorPool)
      if matchingIndex > 0 then
         local tensorIdx = availableTensors[matchingIndex]
         local poolTensor = tensorPool[tensorIdx]
         table.remove(availableTensors, matchingIndex) -- typically the last element
         local poolStorage = poolTensor:storage()
         if outputTensor:storage():size() > poolStorage:size() then
            -- We don't care about the data in the pool tensor, so resize it to zero before growing to avoid a realloc/copy.
            poolStorage:resize(0)
         end
         local viewIdx = #tensorPoolViews + 1
         symbols[output.source] = "vlocals[" .. viewIdx .. "]"
         tensorPoolViews[viewIdx] = outputTensor.new(poolStorage):resize(outputTensor:size())
      else
         -- No match anywhere, allocate new slot in the tensor pool.
         local tensorIdx = #tensorPool + 1
         tensorPool[tensorIdx] = outputTensor.new(outputTensor:size())
         symbols[output.source] = "rlocals[" .. tensorIdx .. "]"
      end
   end

   -- Map aliased outputs.
   for node, aliasNode in pairs(aliases) do
      if not symbols[aliasNode.outputs[1].source] then
         error("invalid node alias")
      end
      symbols[node.outputs[1].source] = symbols[aliasNode.outputs[1].source]
   end

   for i = 1, #graph.mutationFlow.history do
      local aliasOp = graph.mutationFlow.history[i]
      symbols[aliasOp.to.source] = symbols[aliasOp.from.source]
   end

   return symbols, undefined, constants, tensorPoolViews
end

local function collectObjects(execOrder)
   -- Find all the nn objects we need to create or pass in.
   local objects = { }
   local count = 1
   for i = 1, #execOrder do
      local node = execOrder[i]
      local obj = node.forwardFn.object
      if obj ~= nil and objects[obj] == nil then
         if node.forwardFn.ctor then
            objects[obj] = {
               ctor = node.forwardFn.package .. "." .. node.forwardFn.ctor,
               name = string.lower(node.forwardFn.ctor .. count),
               args = node.forwardFn.args
            }
         else
            objects[obj] = {
               object = obj,
               name = node.forwardFn.name .. count
            }
         end
         count = count + 1
      end
   end
   return objects
end

local function changeToReuseFunctions(execOrder)
   for i = 1, #execOrder do
      local node = execOrder[i]
      local tfn = reusableFunctionTransforms[node.forwardFn.name]
      if tfn ~= nil and  #node.outputs == 1 and node.outputs[1].type == Value.TENSOR then
         node.forwardFn.name = tfn
      end
   end
end

local function aliasFreeTensors(execOrder, aliases, hazardNodes)
   local availableTensorMap = { }
   local availableCount = 0
   local refCounts = { }
   local freeTensors = { }
   for i = 1, #execOrder do
      local node = execOrder[i]
      if canReuseOutput(node) then
         refCounts[node.outputs[1]] = #node.outputTargets[1]
         if aliases[node] == nil and hazardNodes[node] == nil then
            if availableCount > 0 then
               local sig = tensorSig(node.outputs[1]:get())
               local matchingList = availableTensorMap[sig]
               if matchingList ~= nil and #matchingList > 0 then
                  local aliasInput = table.remove(matchingList, #matchingList)
                  local target = aliasInput.source:getRoot().node
                  if aliases[target] ~= nil then
                     aliases[node] = aliases[target]
                  else
                     aliases[node] = target
                  end
                  availableCount = availableCount - 1
               end
            end
         end
      end
      for k = 1, #node.inputs do
         local input = node.inputs[k]
         if input.type == Value.TENSOR then
            local refCount = refCounts[input]
            if refCount ~= nil then
               refCounts[input] = refCount - 1
               if refCount == 1 then
                  local sig = tensorSig(input:get())
                  local list = availableTensorMap[sig]
                  if list == nil then
                     list = { }
                     availableTensorMap[sig] = list
                  end
                  list[#list + 1] = input
                  availableCount = availableCount + 1
               end
            end
         end
      end
   end
end

local function addNodeTargets(node, hazardNodes)
   local targetArray = node.outputTargets[1]
   for k = 1, #targetArray do
      local target = targetArray[k]
      local targetNode = target.node
      hazardNodes[targetNode] = true
   end
end

local function generateCode(graph, opt)
   local optimize = opt.optimize or true
   local withForward = util.defaultBool(opt.withForward, true)
   local withGradients = util.defaultBool(opt.withGradients, true)
   local inline = util.defaultBool(opt.inline, true)
   local tensorPool = opt.tensorPool or { }
   local tensorLocals = opt.tensorLocals or { }
   local debugger = opt.debugger

   local execOrder, hazardNodes = graph:walkExecutionOrder(withForward, withGradients)

   -- Don't allow any reordering or inlining of operations near assignment flow.
   for i = 1, #graph.mutationFlow.history do
      local aliasOp = graph.mutationFlow.history[i]
      addNodeTargets(aliasOp.from.source.node, hazardNodes)
      addNodeTargets(aliasOp.to.source.node, hazardNodes)
      hazardNodes[aliasOp.from.source.node] = true
      hazardNodes[aliasOp.to.source.node] = true
   end

   changeToReuseFunctions(execOrder)
   local params = collectParams(graph.params)
   local aliases = { }
   if opt.reduceFootprint then
      if opt.stableGradients then
         aliasFreeTensors(execOrder, aliases, hazardNodes, graph.mutationFlow)
      else
         aliasFreeTensors(execOrder, aliases, { })
      end
   end
   local symbols, undefined, constants, tensorPoolViews = createSymbolTable(graph, execOrder, aliases, params, tensorPool, tensorLocals, opt)
   local objects = collectObjects(execOrder)

   local out = StringBuilder()
   local outerArgNames = {"locals", "rlocals", "vlocals"}
   local outerArgs = { tensorLocals, tensorPool, tensorPoolViews }


   if debugger then
      debugger.setMain(symbols, graph.grads, graph.answers)
      outerArgNames[#outerArgNames + 1] = "debugger"
      outerArgs[#outerArgs + 1] = debugger
   end

   for k, v in pairs(objects) do
      if v.ctor == nil then
         outerArgNames[#outerArgNames + 1] = v.name
         outerArgs[#outerArgs + 1] = k
      end
   end

   local functionRemap = { }
   for i = 1, #execOrder do
      local node = execOrder[i]
      if node.forwardFn.name == "Value.__internal_get" then
         node.forwardFn = { operator = "index", name = "op.__index" }
      end
      if node.forwardFn.name == "Value.__internal_set" then
         node.forwardFn = { operator = "newindex", name = "op.__newindex" }
      end
      if node.forwardFn.operator == nil and functionRemap[node.forwardFn.name] == nil then
         functionRemap[node.forwardFn.name] = string.gsub(node.forwardFn.name, "%.", "_")
      end
   end

   local state = {
      symbols = symbols,
      hazardNodes = hazardNodes,
      functionRemap = functionRemap,
      debugger = debugger,
      inline = inline,
      objects = objects
   }

   -- Generate code.
   out.write("return function(", table.concat(outerArgNames, ", "), ")")
   out.write("\n")
   out.write("local nn = require('autograd').nn")
   out.write("\n")
   out.write("local util = require('autograd.util')")
   out.write("\n")
   for k, v in pairs(objects) do
      if v.ctor ~= nil then
         out.write("local ", v.name, " = ", v.ctor, "(", table.concat(v.args, ", "), ")")
         out.write("\n")
      end
   end
   for k, v in pairs(functionRemap) do
      out.write("local ", v, " = ", k)
      out.write("\n")
   end
   for i = 1, #constants do
      out.write("local ", constants[i]:symbolPath(symbols), " = ", constants[i]:symbolPath({}))
      out.write("\n")
   end
   out.write("return function(")
   local paramSymbols = { }
   for i = 1, #params do
      paramSymbols[i] = symbols[params[i]]
   end
   out.write(table.concat(paramSymbols, ", "))
   out.write(")")
   out.write("\n")
   if debugger then
      for i = 1, #graph.params do
         debugger.generateInputCheck(graph.params[i], paramSymbols[i], out)
      end
   end
   for i = 1, #execOrder do
      local node = execOrder[i]
      local outputSymbols = { }
      for k = 1, #node.outputs do
         outputSymbols[k] = symbols[node.outputs[k].source]
      end
      if not canInline(node, state) then
         out.write("    ")
         if (not canReuseOutput(node)) and (node.forwardFn.operator ~= "newindex") then
            if #outputSymbols > 0 then
               if undefined[node.outputs[1].source] then
                  out.write("local ")
               end
               out.write(table.concat(outputSymbols, ", "), " = ")
            end
         end
         out.write(writeExpr(state, node))
         out.write("\n")
         if debugger then
            for k = 1, #node.outputs do
               debugger.generateOutputCheck(node, k, outputSymbols[k], out)
            end
         end
      end
   end
   out.write("    ")
   out.write("return ")
   local grads = graph.grads
   local answers = graph.answers
   if withGradients then
      if #grads == 1 and grads[1].grad.type == Value.TABLE then
         -- This doesn't feel quite right, should be possible to unify this with the other path.
         out.write(grads[1].grad.source:symbolPath(symbols))
      elseif #grads == 1 and grads[1].grad.type == Value.TENSOR and grads[1].param.source.type == Source.PARAM then
         out.write(grads[1].grad.source:symbolPath(symbols))
      else
         local retTable = { }
         for i = 1, #grads do
            local valTable = retTable
            local stack = grads[i].param.source:getParentsArray()
            local gradSymbol = grads[i].grad.source:symbolPath(symbols)
            for k = 1, #stack do
               local ss = stack[k]
               if ss.type == Source.TABLE then
                  if valTable[ss.key] == nil then
                     if stack[k + 1] == nil then
                        valTable[ss.key] = gradSymbol
                     else
                        local nextTable = { }
                        valTable[ss.key] = nextTable
                     end
                  end
                  valTable = valTable[ss.key]
               end
            end
         end
         writeLiteralTable(retTable, out, symbols, 2)
      end
   end
   if withForward then
      if withGradients then
         out.write(", ")
      end
      for i = 1, #answers do
         if i ~= 1 then
            out.write(", ")
         end
         local answer = answers[i]
         if Value.isValue(answer) then
            out.write(answers[i].source:symbolPath(symbols))
         elseif type(answer) == "table" then
            writeLiteralTable(answer, out, symbols, 2)
         end
      end
   end
   out.write("\n")
   out.write("end")
   out.write("\n")
   out.write("end")
   out.write("\n")
   local code = out.finish()
   if debugger then
      debugger.setCode(code)
   end
   local retValues = { }
   if withGradients then
      retValues = { Value.flattenGrads(graph.params[opt.argnum], graph.intermediateGrads) }
   end
   for i = 1, #graph.answers do
      retValues[#retValues + 1] = flattenAnswer(graph.answers[i])
   end
   return code, outerArgs, retValues
end

local function generateFn(graph, opt)
   local code, outerArgs, retValues = generateCode(graph, opt)
   local outer = (loadstring or load)(code)
   if outer == nil then
      print(code)
      error("failed to parse generated code")
   end
   return outer()(table.unpack(outerArgs)), retValues, code
end

return {
   generateFn = generateFn
}