local overload = require 'autograd.overload'
local Node = require 'autograd.Node'
local Value = require 'autograd.Value'
local Source = require 'autograd.Source'
local StringBuilder = require 'autograd.StringBuilder'
local Debugger = require 'autograd.Debugger'
local DirectTape = require 'autograd.direct.DirectTape'

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
   ["util.sigmoidInPlace"] = true,
   ["util.narrowSliceCopyInPlace"] = true,
   ["util.selectSliceCopyInPlace"] = true,
   ["util.fillSameSizeAsInPlace"] = true,
   ["util.fillSameSizeAsInPlace"] = true,
   ["util.zerosLikeInPlace"] = true,
   ["util.setNotEqualInPlace"] = true,
   ["util.narrowCopyInPlace"] = true,
   ["util.selectCopyInPlace"] = true,
   ["util.indexAddInPlace"] = true,
   ["util.newTensorLikeInPlace"] = true,
}

local reusableFunctionTransforms = {
   ["util.narrowSliceCopy"] = "util.narrowSliceCopyInPlace",
   ["util.selectSliceCopy"] = "util.selectSliceCopyInPlace",
   ["util.fillSameSizeAs"] = "util.fillSameSizeAsInPlace",
   ["util.zerosLike"] = "util.zerosLikeInPlace",
   ["util.setNotEqual"] = "util.setNotEqualInPlace",
   ["util.narrowCopy"] = "util.narrowCopyInPlace",
   ["util.selectCopy"] = "util.selectCopyInPlace",
   ["util.indexAdd"] = "util.indexAddInPlace",
   ["util.sigmoid"] = "util.sigmoidInPlace",
   ["util.newTensorLike"] = "util.newTensorLikeInPlace",
}

local function canReuseOutput(node)
   return reusableFunctionsMap[node.forwardFn.name] ~= nil and #node.outputs == 1 and node.outputs[1].type == Value.TENSOR
end

local function canInline(node, outputNodes, debugger)
   return #node.outputs == 1 and #node.outputTargets[1] == 1 and outputNodes[node] == nil and debugger == nil
end

local function writeExpr(state, node, debugger)
   local out = StringBuilder()
   local inputSymbols = { }
   for k = 1, #node.inputs do
      local input = node.inputs[k]
      if input.source.type == Source.COMPUTED and canInline(input.source.node, state.outputNodes, debugger) then
         local subExpr = writeExpr(state, input.source.node, debugger)
         inputSymbols[k] = "(" .. subExpr .. ")"
      else
         local symbol = input.source:symbolPath(state.symbols)
         inputSymbols[k] = symbol
      end
   end
   if node.forwardFn.operator ~= nil then
      local op = node.forwardFn.operator
      if op == "unm" then
         out.write("-", inputSymbols[1])
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

local function letterForType(val)
   if val.type == Value.TENSOR then
      return "t"
   elseif val.type == Value.NUMBER then
      return "n"
   else
      return "r"
   end
end


local function defaultBool(b, db)
   if b == nil then
      return db
   end
   return b
end

local nodeDebugger
local applyDepth = 0
local nodeDisabled = true

local function nodeCompute(fn, gradFn, capture, ...)
   local inputs = {...}
   applyDepth = applyDepth + 1
   if not nodeDisabled and applyDepth == 1 and capture then
      local n = Node.new(fn, gradFn, inputs)
      local values = {n:evaluateForward()}
      if nodeDebugger then
         nodeDebugger.captureCallStack(n)
      end
      applyDepth = applyDepth - 1
      return table.unpack(values)
   else
      local evalArgs = { }
      for i = 1, #inputs do
         if Value.isValue(inputs[i]) then
            evalArgs[i] = inputs[i]:flatten()
         else
            evalArgs[i] = inputs[i]
         end
      end
      local values = {fn.fn(table.unpack(evalArgs))}
      applyDepth = applyDepth - 1
      return table.unpack(values)
   end
end

local function collectGradients(val, grads)
   grads = grads or { }
   if Value.isValue(val) then
      if val.source.gradients ~= nil then
         for i = 1, #val.source.gradients do
            grads[#grads + 1] = {
               param = val,
               grad = val.source.gradients[i]
            }
         end
      end
      if val.type == Value.TABLE then
         collectGradients(val:get(), grads)
      end
   elseif type(val) == "table" then
      for k, v in pairs(val) do
         collectGradients(v, grads)
      end
   end
   return grads
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

local function convertOperators(execOrder)
   for i = 1, #execOrder do
      local node = execOrder[i]
      if node.forwardFn.operator ~= nil then
         local op = node.forwardFn.operator
         if op == "mul" and #node.inputs == 2 then
            if node.inputs[1].type == Value.TENSOR and node.inputs[2].type == Value.TENSOR then
               local d1 = node.inputs[1].raw:nDimension()
               local d2 = node.inputs[2].raw:nDimension()
               if d1 == 2 and d2 == 2 then
                  node.forwardFn = { name = "torch.mm" }
               elseif d1 == 2 and d2 == 1 then
                  node.forwardFn = { name = "torch.mv" }
               elseif d1 == 1 and d2 == 1 then
                  node.forwardFn = { name = "torch.dot" }
               end
            elseif node.inputs[1].type == Value.TENSOR and node.inputs[2].type == Value.NUMBER then
               node.forwardFn = { name = "torch.mul" }
            elseif node.inputs[1].type == Value.NUMBER and node.inputs[2].type == Value.TENSOR then
               node.forwardFn = { name = "torch.mul" }
               node.inputs[1].source:changeNodeTargetIndex(node, 1, 2)
               node.inputs[2].source:changeNodeTargetIndex(node, 2, 1)
               local t1 = node.inputs[1]
               node.inputs[1] = node.inputs[2]
               node.inputs[2] = t1
            end
         elseif op == "add" and #node.inputs == 2 then
            if node.inputs[1].type == Value.TENSOR and node.inputs[2].type == Value.TENSOR then
               node.forwardFn = { name = "torch.add" }
            elseif node.inputs[1].type == Value.TENSOR and node.inputs[2].type == Value.NUMBER then
               node.forwardFn = { name = "torch.add" }
            elseif node.inputs[1].type == Value.NUMBER and node.inputs[2].type == Value.TENSOR then
               node.forwardFn = { name = "torch.add" }
            end
         elseif op == "unm" then
            if node.inputs[1].type == Value.TENSOR then
               node.forwardFn = { name = "torch.neg" }
            end
         end
      end
   end
end

local function replaceNode(nodeValue, withNodeValue, outputNodes)
   local node = nodeValue.source.node
   node:unlinkInputs()
   local toRemove = { }
   for k = 1, #node.outputs do
      for i = 1, #node.outputTargets[k] do
         toRemove[#toRemove + 1] = node.outputTargets[k][i].node
      end
   end
   for i = 1, #toRemove do
      toRemove[i]:replaceInput(nodeValue, withNodeValue)
   end
   local rootValues = outputNodes[node]
   if rootValues ~= nil then
      for i = 1, #rootValues do
         if rootValues[i].source.type == Source.TABLE then
            rootValues[i].source:changeRoot(withNodeValue.source)
         else
            rootValues[i].source = withNodeValue.source
         end
      end
      outputNodes[replaceNode] = rootValues
      outputNodes[node] = { }
   end
end

local function removeIdentityOperators(execOrder, outputNodes)
   for i = 1, #execOrder do
      local node = execOrder[i]
      if outputNodes[node] == nil then
         local op = node.forwardFn.operator
         if node.forwardFn.operator ~= nil then
            if op == "mul" then
               if node.inputs[1].source.type == Source.CONSTANT and node.inputs[1]:get() == 1 then
                  replaceNode(node.outputs[1], node.inputs[2], outputNodes)
               elseif node.inputs[2].source.type == Source.CONSTANT and node.inputs[2]:get() == 1 then
                  replaceNode(node.outputs[1], node.inputs[1], outputNodes)
               end
            elseif op == "add" or op == "sub" then
               if node.inputs[1].source.type == Source.CONSTANT and node.inputs[1]:get() == 0 then
                  replaceNode(node.outputs[1], node.inputs[2], outputNodes)
               elseif node.inputs[2].source.type == Source.CONSTANT and node.inputs[2]:get() == 0 then
                  replaceNode(node.outputs[1], node.inputs[1], outputNodes)
               end
            end
         end
      end
   end
end

local function convertSubtract(execOrder, outputNodes)
   for i = 1, #execOrder do
      local node = execOrder[i]
      local op = node.forwardFn.operator
      if op == "sub" and #node.inputs == 2 then
         local unmNode = Node.new({ fn = function(a) return -a end, operator = "unm", name = "op.unm" }, nil, { node.inputs[2] })
         local unmOutput = unmNode:evaluateForward()
         local addNode = Node.new({ fn = function(a, b) return a + b end, operator = "add", name = "op.add" }, nil, { node.inputs[1], unmOutput })
         local addOutput = addNode:evaluateForward()
         replaceNode(node.outputs[1], addOutput, outputNodes)
         execOrder[i] = addNode
         table.insert(execOrder, i, unmNode)
      end
   end
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

local function pruneOutputs(execOrder, outputNodes)
   for i = 1, #execOrder do
      local node = execOrder[i]
      if outputNodes[node] == nil then
         for k = #node.outputs, 2, -1 do
            if #node.outputTargets[k] == 0 then
               table.remove(node.outputs, k)
            else
               break
            end
         end
      end
   end
end

local function walkNode(node, order, seen)
   if seen[node] == nil then
      seen[node] = true
      for k = 1, #node.inputs do
         local input = node.inputs[k]
         if input.type == Value.TABLE then
            for k, v in pairs(input:get()) do
               local root = v.source:getRoot()
               if root.type == Source.COMPUTED then
                  walkNode(root.node, order, seen)
               end
            end
         else
            local root = input.source:getRoot()
            if root.type == Source.COMPUTED then
               walkNode(root.node, order, seen)
            end
         end
      end
      table.insert(order, node)
   end
end

local function walkOutputRoots(val, execOrder, seen, outputNodes)
   seen = seen or { }
   execOrder = execOrder or { }
   if Value.isValue(val) then
      local root = val.source:getRoot()
      if root.type == Source.COMPUTED then
         if outputNodes ~= nil then
            local valueList = outputNodes[root.node]
            if outputNodes[root.node] == nil then
               valueList = { }
               outputNodes[root.node] = valueList
            end
            valueList[#valueList + 1] = val
         end
         walkNode(val.source:getRoot().node, execOrder, seen)
      end
   elseif type(val) == "table" then
      for k, subVal in pairs(val) do
         walkOutputRoots(subVal, execOrder, seen, outputNodes)
      end
   end
   return execOrder
end

local function execGraph(graph, withForward, withGradients)
   local seen = { }
   local grads = graph.grads
   local answers = graph.answers
   local execOrder = { }
   local outputNodes = { }
   if defaultBool(withGradients, true) then
      for i = 1, #grads do
         walkOutputRoots(grads[i].grad, execOrder, seen, outputNodes)
      end
   end
   if defaultBool(withForward, true) then
      walkOutputRoots(answers, execOrder, seen, outputNodes)
   end
   return execOrder, outputNodes
end

local function createGraph(fn, args, opt, debugger)
   local argnum = opt.argnum or 1
   local partialGrad = defaultBool(opt.partialGrad, false)
   local withGradients = defaultBool(opt.withGradients, true)
   local values = { }
   local tensorDims = { }

   for i = 1, #args do
      -- Don't wrap the outer tables in Values, since it would interfere with the use of the # operator.
      -- This creates some problems when referring to an entire param table in the generated code - it'll
      -- be represented as a new literal table, but it's a good tradeoff until we move entirely to Lua 5.2
      -- and can overload # on Value.
      values[i] = Value.from(args[i], Source.param(i, i == argnum), true)
   end

   -- Begin recording all torch operations.
   overload.install(nodeCompute)
   nodeDisabled = false
   nodeDebugger = debugger

   -- Call user forward function.
   local answers = {fn(table.unpack(values))}

   -- Figure out forward graph traversal order.
   -- Only walk from the answer we need to differentiate (usually the first).
   local forwardExecOrder = walkOutputRoots(answers[argnum])

   if withGradients then
      -- Walk the execution order backwards, chaining derivatives.
      if answers[1].type == Value.TENSOR and opt.partialGrad then
         answers[1].source.node.gradients[1] = values[#values]
      elseif answers[1].type == Value.NUMBER then
         answers[1].source.node.gradients[1] = Value.from(1, Source.gradient(1))
      else
         error("invalid return value type from autograd function, autograd only supports scalar return values")
      end

      for i=#forwardExecOrder,1,-1 do
         local node = forwardExecOrder[i]
         node:evaluateBackward()
      end
   end

   -- End recording.
   nodeDebugger = nil
   nodeDisabled = true
   overload.uninstall()

   -- Now we have the full graph, forward and backward, determine final traversal order.
   local execOrder = { }
   local seen = { }
   local grads = { }
   local outputNodes = { }

   local grads = collectGradients(values[argnum])

   local graph = {
      grads = grads,
      params = values,
      answers = answers
   }

   return graph
end

local function optimizeGraph(graph, opt)
   local execOrder, outputNodes = execGraph(graph)
   convertSubtract(execOrder, outputNodes)
   removeIdentityOperators(execOrder, outputNodes)
   convertOperators(execOrder)
   changeToReuseFunctions(execOrder)
   pruneOutputs(execOrder, outputNodes)
end

local function tensorSig(t)
   return t:type() .. table.concat(t:size():totable(), ",")
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
         local paramSource = findParamSource(v, params, seen, whichParam)
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

local function createSymbolTable(graph, execOrder, params, tensorPool, tensorLocals)
   -- Assign symbols to params, inputs, outputs.
   local symbols = { }
   local undefined = { }
   local constants = { }

   for i = 1, #params do
      symbols[params[i]] = "p" .. i
   end

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

   -- Exact matches first.
   for i = 1, #execOrder do
      local node = execOrder[i]
      if #node.outputs == 1 then
         local output = node.outputs[1]
         if node.outputs[1].type == Value.TENSOR then
            if canReuseOutput(node) then
               local tensor = output:get()
               local sig = tensorSig(tensor)
               local matchingList = availableTensorMap[sig]
               local tensorIdx = nil
               if matchingList ~= nil and #matchingList > 0 then
                  -- Map to tensor pool.
                  tensorIdx = table.remove(matchingList, #matchingList)
                  availableCount = availableCount - 1
               else
                  if availableCount > 0 then
                     -- There are tensors remaining, so keep track for possible later inexact allocation.
                     remainingOutputs[#remainingOutputs + 1] = i
                  else
                     -- No available tensors, so just go ahead and allocate a slot for this one now.
                     tensorIdx = #tensorPool + 1
                     tensorPool[tensorIdx] = tensor.new(tensor:size())
                  end
               end
               if tensorIdx ~= nil then
                  symbols[output.source] = "rlocals[" .. tensorIdx .. "]"
               end
            else
               -- Non-reusable local.
               localCount = localCount + 1
               tensorLocals[localCount] = 0
               symbols[output.source] = "locals[" .. localCount .. "]"
            end
         else
            -- One output, not a tensor.
            undefined[output.source] = true
            symbols[output.source] = letterForType(node.outputs[1]) .. i
         end
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
         if source.type == Source.CONSTANT and symbols[source] == nil and torch.isTensor(source.val) then
            local index = constantTensorMap[source.val]
            if index == nil then
               index = #constants + 1
               constantTensorMap[source.val] = index
               constants[index] = source
            end
            symbols[source] = "c" .. index
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

local function generateCode(fn, args, opt)
   local optimize = opt.optimize or true
   local withForward = defaultBool(opt.withForward, true)
   local withGradients = defaultBool(opt.withGradients, true)
   local tensorPool = opt.tensorPool or { }
   local tensorLocals = opt.tensorLocals or { }
   local debugger
   if opt.debugHook then
      debugger = Debugger(opt)
   end

   local graph = createGraph(fn, args, opt, debugger)

   if optimize then
      optimizeGraph(graph)
   end

   local execOrder, outputNodes = execGraph(graph, withForward, withGradients)
   local params = collectParams(graph.params)
   local symbols, undefined, constants, tensorPoolViews = createSymbolTable(graph, execOrder, params, tensorPool, tensorLocals)
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
      if node.forwardFn.operator == nil and functionRemap[node.forwardFn.name] == nil then
         functionRemap[node.forwardFn.name] = string.gsub(node.forwardFn.name, "%.", "_")
      end
   end

   local state = {
      symbols = symbols,
      outputNodes = outputNodes,
      functionRemap = functionRemap,
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
      if not canInline(node, outputNodes, debugger) then
         out.write("    ")
         if not canReuseOutput(node) then
            if #outputSymbols > 0 then
               if undefined[node.outputs[1].source] then
                  out.write("local ")
               end
               out.write(table.concat(outputSymbols, ", "), " = ")
            end
         end
         out.write(writeExpr(state, node, debugger))
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
   local retValues = { Value.flattenGrads(graph.params[opt.argnum]) }
   for i = 1, #graph.answers do
      retValues[#retValues + 1] = flattenAnswer(graph.answers[i])
   end
   return code, outerArgs, retValues
end

local function execUncached(fn, args, opt)
   local graph = createGraph(fn, args, opt, debugger)
   local retValues = { Value.flattenGrads(graph.params[opt.argnum]) }
   for i = 1, #graph.answers do
      retValues[#retValues + 1] = flattenAnswer(graph.answers[i])
   end
   return table.unpack(retValues)
end

local function buildSignature(params, tensorDims)
   for k, v in pairs(params) do
      if torch.isTensor(v) then
         tensorDims[#tensorDims + 1] = table.concat(v:size():totable(), "x")
      elseif type(v) == 'number' then
         tensorDims[#tensorDims + 1] = "n"
      elseif type(v) == 'table' then
         tensorDims[#tensorDims + 1] = "t" .. #v
         buildSignature(v, tensorDims)
      end
   end
end

local defaultOptimize = false

local function optimize(opt)
   defaultOptimize = opt
end

local function printPoolStats(tensorPool)
   local size = 0
   for i = 1, #tensorPool do
      local tensor = tensorPool[i]
      size = size + tensor:storage():size() * 4
   end
   print("tensor pool size: " .. (size / (1024 * 1024)) .. " MB")
end

local function grad(fn, gradOpt)
   gradOpt = gradOpt or { }
   local argnum = gradOpt.gradArg or 1
   local optimize = defaultBool(gradOpt.optimize, defaultOptimize)
   local withForward = defaultBool(gradOpt.withForward, true)
   local withGradients = defaultBool(gradOpt.withGradients, true)
   local partialGrad = defaultBool(gradOpt.partialGrad, false)
   local debugHook = gradOpt.debugHook
   local generatedFunctions = { }
   local tensorPool = { }
   local tensorLocals = { }
   local opt = {
      argnum = argnum,
      withForward = withForward,
      withGradients = withGradients,
      partialGrad = partialGrad,
      debugHook = debugHook,
      tensorPool = tensorPool,
      tensorLocals = tensorLocals
   }
   if optimize then
      local doGrad = function(...)
         local args = {...}
         local tensorDims = { }
         local sigFun = gradOpt.signatureFn or function(params)
            buildSignature(params, tensorDims)
            return table.concat(tensorDims, "-")
         end
         local signature = sigFun(args)
         if signature == nil then
            return execUncached(fn, args, opt)
         end
         if generatedFunctions[signature] == nil then
            local code, outerArgs, retValues = generateCode(fn, args, opt)
            --printPoolStats(tensorPool)
            --print(code)
            --print("generated code for param signature " .. signature)
            local outer = (loadstring or load)(code)
            if outer == nil then
               print(code)
               error("failed to parse generated code")
            end
            generatedFunctions[signature] = outer()(table.unpack(outerArgs))
            -- We already have the answers, don't run it all over again.
            if withGradients and withForward and not debugHook then
               return table.unpack(retValues)
            end
         end
         return generatedFunctions[signature](table.unpack(args))
      end
      return doGrad
   else
      if withForward and withGradients then
         return function(...)
            return DirectTape.grad(fn, argnum, nil, ...)
         end
      elseif withForward then
         return function(...)
            return fn(...)
         end
      elseif withGradients then
         return function(...)
            local args = {...}
            local partialGrad = table.remove(args, #args)
            return DirectTape.grad(fn, argnum, partialGrad, table.unpack(args))
         end
      end
   end
end

-- Support functions
include 'support.lua'

-- Standard overloaded functions with gradients
include 'gradfuns.lua'

-- Main functions:
local autograd = {
   grad = grad,
   overload = overload,
   optimize = optimize
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
