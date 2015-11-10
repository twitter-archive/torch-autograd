local debug = require 'debug'
local overload = require 'autograd.overload'
local Node = require 'autograd.Node'
local Value = require 'autograd.Value'
local Source = require 'autograd.Source'

local reusableFunctions = {
   "torch.tanh",
   "torch.cmul",
   "torch.cdiv",
   "torch.exp",
   "torch.pow",
   "torch.add",
   "torch.mul",
   "torch.neg",
   "torch.mm",
   "torch.mv",
}

local reusableFunctionsMap = { }
for i = 1, #reusableFunctions do
   reusableFunctionsMap[reusableFunctions[i]] = true
end

local function stringBuilder()
   local strs = { }
   return {
      write = function(...)
         local arg = {...}
         for i = 1, #arg do
            strs[#strs + 1] = arg[i]
         end
      end,
      finish = function()
         return table.concat(strs, "")
      end
   }
end

local function walkExecutionOrder(symbols, node, seen, order)
   if seen[node] == nil then
      seen[node] = true
      for k = 1, #node.inputs do
         local input = node.inputs[k]
         local root = input.source:getRoot()
         if root.type == Source.COMPUTED then
            walkExecutionOrder(symbols, root.node, seen, order)
         end
      end
      table.insert(order, node)
   end
end

local function canReuseOutput(node)
   return reusableFunctionsMap[node.forwardFn.name] and #node.outputs == 1 and node.outputs[1].type == Value.TENSOR
end

local function canInline(node, outputNodes)
   return #node.outputs == 1 and #node.outputTargets[1] == 1 and outputNodes[node] == nil
end

local function writeExpr(state, node, depth)
   local out = stringBuilder()
   local inputSymbols = { }
   for k = 1, #node.inputs do
      local input = node.inputs[k]
      if input.source.type == Source.COMPUTED and canInline(input.source.node, state.outputNodes) then
         local subExpr = writeExpr(state, input.source.node)
         inputSymbols[k] = "(" .. subExpr .. ")"
      else
         local symbol = input.source:symbolPath(state.symbols)
         inputSymbols[k] = symbol
      end
      --[[
      if state.refCount[input.source] ~= nil then
         state.refCount[input.source] = state.refCount[input.source] - 1
      end
      --]]
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
      if canReuseOutput(node) then
         table.insert(inputSymbols, 1, node.outputs[1].source:symbolPath(state.symbols))
      end
      out.write(state.functionRemap[node.forwardFn.name], "(", table.concat(inputSymbols, ", "), ")")
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

local applyDepth = 0

local function nodeCompute(fn, gradFn, capture, ...)
   local inputs = {...}
   applyDepth = applyDepth + 1
   if applyDepth == 1 and capture then
      local n = Node.new(fn, gradFn, inputs)
      local values = {n:evaluateForward()}
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

local function findGradients(val, grads)
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
         findGradients(v, grads)
      end
   end
end

local function writeLiteralTable(wtable, out, depth)
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
      if type(v) == 'table' then
         writeLiteralTable(v, out, depth + 1)
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

local function removeIdentityOperators(execOrder)
   for i = 1, #execOrder do
      local node = execOrder[i]
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

local function generateCode(fn, args, argnum)
   local values = { }
   local tensorDims = { }
   for i = 1, #args do
      values[i] = Value.from(args[i], Source.param(i, i == argnum))
   end

   overload.install(nodeCompute)

   -- Call user forward function
   local answers = {fn(unpack(values))}

   -- Figure out graph traversal order.
   local seen = { }
   local forwardExecOrder = { }
   walkExecutionOrder(symbols, answers[1].source.node, seen, forwardExecOrder)

   -- Walk forward-graph backwards, chaining derivatives.
   answers[1].source.node.gradients[1] = Value.from(1, Source.gradient(1))
   for i=#forwardExecOrder,1,-1 do
      local node = forwardExecOrder[i]
      node:evaluateBackward()
   end

   overload.uninstall()

   -- Now we have the full graph, forward and backward, determine final traversal order.
   local execOrder = { }
   local seen = { }
   local grads = { }
   local outputNodes = { }

   findGradients(values[argnum], grads)
   for i = 1, #grads do
      walkExecutionOrder(symbols, grads[i].grad.source:getRoot().node, seen, execOrder)
      outputNodes[grads[i].grad.source:getRoot().node] = true
   end

   local localTable = true

   removeIdentityOperators(execOrder)
   convertOperators(execOrder)

   -- Re-evaluate exec order after optimizations.
   seen = { }
   execOrder = { }
   for i = 1, #grads do
      walkExecutionOrder(symbols, grads[i].grad.source:getRoot().node, seen, execOrder)
   end

   -- Assign symbols to params, inputs, outputs.
   local symbols = { }
   local refCount = { }
   local functionRemap = { }

   for i = 1, #values do
      symbols[values[i].source] = "p" .. i
   end

   for i = 1, #execOrder do
      local node = execOrder[i]
      if #node.outputs == 1 then
         if localTable then
            symbols[node.outputs[1].source] = "locals[" .. i .. "]"
         else
            symbols[node.outputs[1].source] = letterForType(node.outputs[1]) .. i
         end
      else
         for k = 1, #node.outputs do
            local output = node.outputs[k]
            if localTable then
               error('todo')
               symbols[output.source] = ""
            else
               symbols[node.outputs[k].source] = letterForType(node.outputs[k]) .. i
            end
         end
      end
      --[[
      if outputNodes[node] == nil then
         for k = 1, #node.outputs do
            local output = node.outputs[k]
            if output.type == Value.TENSOR then
               refCount[output.source] = #node.outputTargets[k]
            end
         end
      end
      --]]
   end

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

   -- Collect the objects we use, where we couldn't determine how to construct
   -- them. They have to be passed in by value.
   local out = stringBuilder()
   local noCtorObjectNames = { }
   local outerArgs = { }
   for k, v in pairs(objects) do
      if v.ctor == nil then
         noCtorObjectNames[#noCtorObjectNames + 1] = v.name
         outerArgs[#outerArgs + 1] = k
      end
   end

   for i = 1, #execOrder do
      local node = execOrder[i]
      if node.forwardFn.operator == nil then
         functionRemap[node.forwardFn.name] = string.gsub(node.forwardFn.name, "%.", "_")
      end
   end

   local state = {
      symbols = symbols,
      outputNodes = outputNodes,
      refCount = refCount,
      functionRemap = functionRemap,
      objects = objects
   }

   -- Generate code.
   out.write("return function(", table.concat(noCtorObjectNames, ", "), ")")
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
   if localTable then
      out.write("local locals = { }")
      out.write("\n")
      out.write("for i = 1, ", #execOrder, " do locals[i] = 0 end")
      out.write("\n")
      for i = 1, #execOrder do
         local node = execOrder[i]
         if canReuseOutput(node) then
            local tensor = node.outputs[1].raw
            out.write(symbols[node.outputs[1].source], " = ", tensor:type(), "(", table.concat(tensor:size():totable(), ", "), ")")
            out.write("\n")
         end
      end
   end

   out.write("return function(")
   local paramSymbols = { }
   for i = 1, #values do
      paramSymbols[i] = symbols[values[i].source]
   end
   out.write(table.concat(paramSymbols, ", "))
   out.write(")")
   out.write("\n")
   for i = 1, #execOrder do
      local node = execOrder[i]
      local outputSymbols = { }
      for k = 1, #node.outputs do
         outputSymbols[k] = symbols[node.outputs[k].source]
      end
      if not canInline(node, outputNodes) then
         out.write("    ")
         if not canReuseOutput(node) then
            if #outputSymbols > 0 then
               if not localTable then
                  out.write("local ")
               end
               out.write(table.concat(outputSymbols, ", "), " = ")
            end
         end
         out.write(writeExpr(state, node))
         out.write("\n")
         -- local toFree = { }
         -- for k, v in pairs(refCount) do
         --    if v == 0 then
         --       toFree[#toFree + 1] = k
         --    end
         -- end
         -- for k = 1, #toFree do
         --    refCount[toFree[k]] = nil
         --    out.write("    ")
         --    out.write(symbols[toFree[k]])
         --    out.write(":free()")
         --    out.write("\n")
         -- end
      end
   end
   out.write("    ")
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
   out.write("return ")
   writeLiteralTable(retTable, out, 2)
   out.write("\n")
   out.write("end")
   out.write("\n")
   out.write("end")
   out.write("\n")
   return out.finish(), outerArgs
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

local function grad(fn, argnum)
   argnum = argnum or 1
   local generatedFunctions = { }
   local doGrad = function(...)
      local args = {...}
      local tensorDims = { }
      buildSignature(args, tensorDims)
      local signature = table.concat(tensorDims, "-")
      if generatedFunctions[signature] == nil then
         local code, outerArgs = generateCode(fn, args, argnum)
        --print(code)
        --print("generated code for param signature " .. signature)
         generatedFunctions[signature] = loadstring(code)()(unpack(outerArgs))
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
