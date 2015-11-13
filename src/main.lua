local overload = require 'autograd.overload'
local Node = require 'autograd.Node'
local Value = require 'autograd.Value'
local Source = require 'autograd.Source'
local StringBuilder = require 'autograd.StringBuilder'
local Debugger = require 'autograd.Debugger'

local reusableFunctionsMap = {
   ["torch.tanh"] = true,
   ["torch.cmul"] = true,
   ["torch.cdiv"] = true,
   ["torch.exp"]  = true,
   ["torch.pow"]  = true,
   ["torch.add"]  = true,
   ["torch.mul"]  = true,
   ["torch.neg"]  = true,
   ["torch.ger"]  = true,
   ["torch.mm"]   = true,
   ["torch.mv"]   = true,
   ["torch.cosh"] = true,
   ["torch.expand"] = true,
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
      return unpack(values)
   else
      local evalArgs = { }
      for i = 1, #inputs do
         if Value.isValue(inputs[i]) then
            evalArgs[i] = inputs[i]:flatten()
         else
            evalArgs[i] = inputs[i]
         end
      end
      local values = {fn.fn(unpack(evalArgs))}
      applyDepth = applyDepth - 1
      return unpack(values)
   end
end

local function collectGradients(val, grads)
   grads = grads or { }
   if val.source.gradients ~= nil then
      for i = 1, #val.source.gradients do
         grads[#grads + 1] = {
            param = val,
            grad = val.source.gradients[i]
         }
      end
   end
   if val.type == Value.TABLE then
      for k, v in pairs(val:get()) do
         collectGradients(v, grads)
      end
   end
   return grads
end

local function writeLiteralTable(wtable, out, symbols, depth)
   depth = depth or 1
   out.write("{", "\n")
   for k, v in pairs(wtable) do
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

local function replaceNode(nodeValue, withNodeValue)
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
end

local function removeIdentityOperators(execOrder, outputNodes)
   for i = 1, #execOrder do
      local node = execOrder[i]
      if outputNodes[node] == nil then
         local op = node.forwardFn.operator
         if node.forwardFn.operator ~= nil then
            if op == "mul" then
               if node.inputs[1].source.type == Source.CONSTANT and node.inputs[1]:get() == 1 then
                  replaceNode(node.outputs[1], node.inputs[2])
               elseif node.inputs[2].source.type == Source.CONSTANT and node.inputs[2]:get() == 1 then
                  replaceNode(node.outputs[1], node.inputs[1])
               end
            elseif op == "add" or op == "sub" then
               if node.inputs[1].source.type == Source.CONSTANT and node.inputs[1]:get() == 0 then
                  replaceNode(node.outputs[1], node.inputs[2])
               elseif node.inputs[2].source.type == Source.CONSTANT and node.inputs[2]:get() == 0 then
                  replaceNode(node.outputs[1], node.inputs[1])
               end
            end
         end
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
            outputNodes[val.source:getRoot().node] = true
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

local function searchMatchingTensorExact(tensor, availableTensors, locals)
   local ttype = tensor:type()
   local tdims = tensor:nDimension()
   for i = #availableTensors, 1, -1 do
      local idx = availableTensors[i]
      local lt = locals[idx]
         if lt:type() == ttype and lt:nDimension() == tdims and lt:isSameSizeAs(tensor) then
            return i
         end
      end
   return 0
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

local function createGraph(fn, args, opt, debugger)
   local argnum = opt.argnum or 1
   local partialGrad = defaultBool(opt.partialGrad, false)
   local withGradients = defaultBool(opt.withGradients, true)
   local values = { }
   local tensorDims = { }

   for i = 1, #args do
      values[i] = Value.from(args[i], Source.param(i, i == argnum))
   end

   -- Begin recording all torch operations.
   overload.install(nodeCompute)
   nodeDisabled = false
   nodeDebugger = debugger

   -- Call user forward function.
   local answers = {fn(unpack(values))}

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
   removeIdentityOperators(execOrder, outputNodes)
   convertOperators(execOrder)
   changeToReuseFunctions(execOrder)
   pruneOutputs(execOrder, outputNodes)
end

local function createSymbolTable(graph, execOrder, reuseLocals)
   -- Assign symbols to params, inputs, outputs.
   local symbols = { }
   local defined = { }
   local constants = { }

   for i = 1, #graph.params do
      symbols[graph.params[i].source] = "p" .. i
   end

   local reusableLocalMap = { }
   local nextReusableLocal = #reuseLocals + 1
   local reusableLocalStart = nextReusableLocal
   local nextLocal = 1
   local availableTensors = { }
   for i = 1, #reuseLocals do
      availableTensors[i] = i
   end

   local tensorAllocations = { }
   local partialMatchAllocations = { }
   local unmatchedTensors = false

   -- Exact matches first.
   for i = #execOrder, 1, -1 do
      local node = execOrder[i]
      if #node.outputs == 1 then
         if node.outputs[1].type == Value.TENSOR then
            if canReuseOutput(node) then
               local matchingIndex = searchMatchingTensorExact(node.outputs[1]:get(), availableTensors, reuseLocals)
               if matchingIndex > 0 then
                  local tensorIdx = availableTensors[matchingIndex]
                  table.remove(availableTensors, matchingIndex) -- typically the last element (or close), since we iterate in reverse
                  tensorAllocations[node.outputs[1]] = tensorIdx
               else
                  unmatchedTensors = true
               end
            end
         end
      end
   end

   if unmatchedTensors and #availableTensors > 0 then
      function sortLocalSize(a, b)
        return reuseLocals[a]:storage():size() < reuseLocals[b]:storage():size()
      end

      table.sort(availableTensors, sortLocalSize)

      local remainingOutputs = { }
   for i = 1, #execOrder do
      local node = execOrder[i]
      if #node.outputs == 1 then
         if node.outputs[1].type == Value.TENSOR then
               if canReuseOutput(node) and tensorAllocations[node.outputs[1]] == nil then
                  remainingOutputs[#remainingOutputs + 1] = node.outputs[1]
               end
            end
         end
      end

      function sortTensorSize(a, b)
        return a:get():storage():size() < b:get():storage():size()
      end

      table.sort(remainingOutputs, sortTensorSize)

      for i = #remainingOutputs, 1, -1 do
         local output = remainingOutputs[i]
         local matchingIndex = searchMatchingTensorLargest(output:get(), availableTensors, reuseLocals)
                  if matchingIndex > 0 then
            local tensorIdx = availableTensors[matchingIndex]
            table.remove(availableTensors, matchingIndex) -- typically the last element
            tensorAllocations[output] = tensorIdx
            partialMatchAllocations[output] = tensorIdx
         end
                  end
               end

   for i = 1, #execOrder do
      local node = execOrder[i]
      if #node.outputs == 1 then
         if node.outputs[1].type == Value.TENSOR then
            defined[node.outputs[1].source] = true
            if canReuseOutput(node) then
               local index = tensorAllocations[node.outputs[1]]
               if index == nil then
                  index = nextReusableLocal
                  nextReusableLocal = nextReusableLocal + 1
               end
               if partialMatchAllocations[node.outputs[1]] ~= nil then
                  symbols[node.outputs[1].source] = "vlocals[" .. index .. "]"
               else
               symbols[node.outputs[1].source] = "rlocals[" .. index .. "]"
               end
               reusableLocalMap[index] = node.outputs[1]
            else
               symbols[node.outputs[1].source] = "locals[" .. nextLocal .. "]"
               nextLocal = nextLocal + 1
            end

         else
            symbols[node.outputs[1].source] = letterForType(node.outputs[1]) .. i
         end
      else
         for k = 1, #node.outputs do
            local output = node.outputs[k]
            symbols[node.outputs[k].source] = letterForType(node.outputs[k]) .. i .. "_" .. k
         end
      end
      for k = 1, #node.inputs do
         local input = node.inputs[k]
         local source = input.source:getRoot()
         if source.type == Source.CONSTANT and symbols[source] == nil and torch.isTensor(source.val) then
            constants[#constants + 1] = source
            symbols[source] = "c" .. #constants
         end
      end
   end
   return symbols, defined, constants, nextLocal - 1, nextReusableLocal - 1, reusableLocalMap, partialMatchAllocations
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
   local reuseLocals = opt.reuseLocals or { }
   local debugger
   if opt.debugHook then
      debugger = Debugger(opt)
   end

   local graph = createGraph(fn, args, opt, debugger)

   if optimize then
      optimizeGraph(graph)
   end

   local execOrder, outputNodes = execGraph(graph, withForward, withGradients)
   local symbols, defined, constants, numLocals, numReusableLocals, reusableLocalMap, partialMatchAllocations = createSymbolTable(graph, execOrder, reuseLocals)
   local objects = collectObjects(execOrder)

   local out = StringBuilder()
   local outerArgNames = {"rlocals"}
   local outerArgs = { }
   outerArgs[#outerArgs + 1] = reuseLocals

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
      if node.forwardFn.operator == nil then
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
   out.write("local locals = { }")
   out.write("local vlocals = { }")
   out.write("\n")
   out.write("for i = ", 1, ", ", numLocals, " do locals[i] = 0 end")
   out.write("\n")
   if numReusableLocals ~= #reuseLocals then
      out.write("for i = ", #reuseLocals + 1, ", ", numReusableLocals, " do rlocals[i] = 0 end")
      out.write("\n")
      for i = #reuseLocals + 1, numReusableLocals do
         local output = reusableLocalMap[i]
         local tensor = output:get()
         out.write(symbols[output.source], " = ", tensor:type(), "(", table.concat(tensor:size():totable(), ", "), ")")
         out.write("\n")
      end
   end
   out.write("for i = ", 1, ", ", numReusableLocals, " do vlocals[i] = 0 end")
   out.write("\n")
   for k, v in pairs(partialMatchAllocations) do
      local output = k
      out.write(symbols[output.source], " = ", output:get():type(), ".new(rlocals[", v, "]):resize(", table.concat(output:get():size():totable(), ", "), ")")
      out.write("\n")
   end
   out.write("\n")
   out.write("return function(")
   local paramSymbols = { }
   for i = 1, #graph.params do
      paramSymbols[i] = symbols[graph.params[i].source]
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
               if not defined[node.outputs[1].source] then
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
   return code, outerArgs
end

local function execUncached(fn, args, opt)
   local graph = createGraph(fn, args, opt, debugger)
   local retValues = { graph.params[opt.argnum]:flattenGrads() }
   for i = 1, #graph.answers do
      retValues[#retValues + 1] = graph.answers[i]:flatten()
   end
   return unpack(retValues)
end

local function buildSignature(params, tensorDims)
   for k, v in pairs(params) do
      if torch.isTensor(v) then
         tensorDims[#tensorDims + 1] = table.concat(v:size():totable(), "x")
      elseif type(v) == 'number' then
         tensorDims[#tensorDims + 1] = "n"
      elseif type(v) == 'table' then
         buildSignature(v, tensorDims)
      end
   end
end

local function grad(fn, gradOpt)
   gradOpt = gradOpt or { }
   local argnum = gradOpt.gradArg or 1
   local withForward = defaultBool(gradOpt.withForward, true)
   local withGradients = defaultBool(gradOpt.withGradients, true)
   local partialGrad = defaultBool(gradOpt.partialGrad, false)
   local debugHook = gradOpt.debugHook
   local generatedFunctions = { }
   local cachedTensors = { }
   local opt = {
      argnum = argnum,
      withForward = withForward,
      withGradients = withGradients,
      partialGrad = partialGrad,
      debugHook = debugHook,
      reuseLocals = cachedTensors
   }
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
         local code, outerArgs = generateCode(fn, args, opt)
         --print(code)
         --print("generated code for param signature " .. signature)
         local outer = loadstring(code)
         if outer == nil then
            error("failed to parse generated code")
         end
         generatedFunctions[signature] = outer()(unpack(outerArgs))
      end
      return generatedFunctions[signature](unpack(args))
   end
   return doGrad
end

-- Support functions
include 'support.lua'

-- Standard overloaded functions with gradients
include 'gradfuns.lua'

-- Sub packages:
local functionalize = (require 'autograd.nnwrapper')(nodeCompute)
local nn = require('autograd.nnwrapper')(nodeCompute)('nn')

-- Main functions:
local autograd = {
   grad = grad,
   overload = overload,
   functionalize = functionalize,
   nn = nn
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
