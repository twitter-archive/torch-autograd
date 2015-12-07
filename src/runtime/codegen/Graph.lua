local Graph = { }

Graph.__index = Graph

local overload = require 'autograd.overload'
local Node = require 'autograd.runtime.codegen.Node'
local Value = require 'autograd.runtime.codegen.Value'
local Source = require 'autograd.runtime.codegen.Source'
local util = require 'autograd.util'

local nodeDebugger
local applyDepth = 0
local nodeDisabled = true

local function overloadHook(fn, gradFn, capture, ...)
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

function Graph:walkExecutionOrder(withForward, withGradients)
   local seen = { }
   local grads = self.grads
   local answers = self.answers
   local execOrder = { }
   local outputNodes = { }
   if util.defaultBool(withGradients, true) then
      for i = 1, #grads do
         walkOutputRoots(grads[i].grad, execOrder, seen, outputNodes)
      end
   end
   if util.defaultBool(withForward, true) then
      walkOutputRoots(answers, execOrder, seen, outputNodes)
   end
   return execOrder, outputNodes
end

function Graph.record(fn, args, opt)
   local argnum = opt.argnum or 1
   local debugger = opt.debugger
   local partialGrad = util.defaultBool(opt.partialGrad, false)
   local withGradients = util.defaultBool(opt.withGradients, true)
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
   overload.install(overloadHook)
   applyDepth = 0
   nodeDisabled = false
   nodeDebugger = debugger

   -- Call user forward function.
   local answers = nil

   local protectedFn = function()
      answers = {fn(table.unpack(values))}

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
   end

   local ok, msg = pcall(protectedFn)

   -- End recording.
   nodeDebugger = nil
   nodeDisabled = true
   overload.uninstall()

   if not ok then
      error(msg)
   end

   local grads = collectGradients(values[argnum])

   local graph = {
      grads = grads,
      params = values,
      answers = answers
   }

   setmetatable(graph, Graph)

   return graph
end

function Graph:optimize()
   local execOrder, outputNodes = self:walkExecutionOrder()
   convertSubtract(execOrder, outputNodes)
   removeIdentityOperators(execOrder, outputNodes)
   convertOperators(execOrder)
   pruneOutputs(execOrder, outputNodes)
end

return Graph