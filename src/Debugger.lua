local Source = require 'autograd.Source'
local Value = require 'autograd.Value'
local StringBuilder = require 'autograd.StringBuilder'

local function Debugger(opt)
   opt = opt or { }
   if not opt.debugHook then
      return nil
   end
   local debugHook = opt.debugHook

   local self

   local debugNodes = { }
   local function debugNode(node)
      if not node.debug then
         table.insert(debugNodes, node)
         node.debug = { index = #debugNodes }
      end
   end

   local function captureCallStack(node)
      debugNode(node)
      local tb = debug.traceback()
      if not tb:match("'evaluateBackward'") then
         node.debug.isForward = true
         local lines = tb:split("\n")
         table.remove(lines, 1) -- Remove the header line
         local infos = { }
         for i,line in ipairs(lines) do
            local info = debug.getinfo(i)
            if info.name == 'generateCode' then
               for j = #infos,1,-1 do
                  if infos[j].what == 'tail' then
                     break
                  end
                  node.debug.callStack = node.debug.callStack or { }
                  table.insert(node.debug.callStack, infos[j])
               end
               break
            else
               table.insert(infos, info)
            end
         end
      end
   end

   local function rcsvFindCallStack(node)
      if node then
         if node.debug and node.debug.callStack then
            return node
         end
         if node.inputs and #node.inputs > 0 then
            return rcsvFindCallStack(node.inputs[1].source.node)
         end
      end
   end

   local function generateComment(node, out)
      local forwardNode = rcsvFindCallStack(node)
      if forwardNode then
         out.write("--[[\n")
         if node == forwardNode then
            out.write("\tFORWARD: (" .. forwardNode.forwardFn.name .. ")\n")
         else
            out.write("\tBACKWARD: (" .. forwardNode.forwardFn.name .. ")\n")
         end
         for i,info in ipairs(forwardNode.debug.callStack) do
            out.write("\t\t" .. i .. ": " .. tostring(info.name) .. info.source .. ":" .. info.currentline .. "\n")
         end
         out.write("--]]\n")
      end
   end

   local function isNanOrInf(x)
      if x ~= x then
         return true
      else
         local s = tostring(x)
         if s == "inf" or s == "-inf" then
            return true
         end
      end
   end

   local function valueMinMax(nodeIndex, outputIndex, value, min, max)
      local node = debugNodes[nodeIndex]
      local output = node.outputs[outputIndex]
      output.debug = output.debug or { }
      output.debug.min = (output.debug.min ~= nil and math.min(output.debug.min, min)) or min
      output.debug.max = (output.debug.max ~= nil and math.max(output.debug.max, max)) or max
      if isNanOrInf(output.debug.min) or isNanOrInf(output.debug.max) then
         debugHook(self, "Detected NaN or Inf")
      end
   end

   local function valueCheckTensor(nodeIndex, outputIndex, value)
      valueMinMax(nodeIndex, outputIndex, value, value:min(), value:max())
   end

   local function valueCheckNumber(nodeIndex, outputIndex, value)
      valueMinMax(nodeIndex, outputIndex, value, value, value)
   end

   local function generateValueCheck(node, outputIndex, symbol, out)
      debugNode(node)
      local output = node.outputs[outputIndex]
      if output.type == Value.TENSOR then
         out.write("    debugger.valueCheckTensor(" .. table.concat({ node.debug.index, outputIndex, symbol }, ", ") .. ")\n")
      elseif output.type == Value.NUMBER then
         out.write("    debugger.valueCheckNumber(" .. table.concat({ node.debug.index, outputIndex, symbol }, ", ") .. ")\n")
      end
   end

   local execOrder
   local symbols
   local function setExecOrderAndSymbols(eo, sy)
      execOrder = eo
      symbols = sy
   end

   local function generateDot(fileName)
      -- Figure out all the variables both supplied and intermediate
      local vars = { }
      local function varKey(io)
         return 'x' .. io.source:symbolPath(symbols):gsub("[^a-zA-Z0-9]+", "_")
      end
      local function addVar(io)
         local key = varKey(io)
         if not vars[key] then
            local parts = { }
            if io.source.type ~= Source.COMPUTED then
               table.insert(parts, io.source:symbolPath(symbols))
            end
            if torch.isTensor(io.raw) then
               table.insert(parts, torch.typename(io.raw):sub(7) .. '(' .. table.concat(io.raw:size():totable(), ", ") .. ')')
            elseif torch.isStorage(io.raw) then
               table.insert(parts, torch.typename(io.raw):sub(7) .. '(' .. io.raw:size() .. ')')
            end
            local color = "black"
            if io.debug and io.debug.min ~= nil then
               table.insert(parts, '[' .. io.debug.min .. ', ' .. io.debug.max .. ']')
               if isNanOrInf(io.debug.min) or isNanOrInf(io.debug.max) then
                  color = "red"
               end
            end
            vars[key] = { label = table.concat(parts, '<BR/>'), color = color }
         end
      end
      for i,node in ipairs(execOrder) do
         for j,input in ipairs(node.inputs) do
            addVar(input)
         end
         for j,output in ipairs(node.outputs) do
            addVar(output)
         end
      end

      -- Build a DOT
      local out = StringBuilder(fileName)
      out.write('digraph graphname {\n')
      for key,var in pairs(vars) do
         out.write('\t' .. key .. ' [label=<' .. var.label .. '> color="' .. var.color .. '" shape=box];\n')
      end
      for i,node in ipairs(execOrder) do
         local label = node.forwardFn.name
         color = "black"
         for _,output in ipairs(node.outputs) do
            if output.debug and (isNanOrInf(output.debug.min) or isNanOrInf(output.debug.max)) then
               color = "red"
            end
         end
         out.write('\tnode' .. i .. ' [label="' .. label .. '" color="'..color..'"];\n')
      end
      for i,node in ipairs(execOrder) do
         local color = (node.debug.isForward and 'green') or 'blue'
         for j,input in ipairs(node.inputs) do
            out.write('\t' .. varKey(input) .. ' -> ' .. 'node' .. i .. ' [color="'..color..'"];\n')
         end
         for j,output in ipairs(node.outputs) do
            out.write('\tnode' .. i .. ' -> ' .. varKey(output) .. ' [color="'..color..'"];\n')
         end
      end
      out.write('}\n')
      out.finish()
   end

   self = {
      captureCallStack = captureCallStack,
      generateComment = generateComment,
      valueCheckTensor = valueCheckTensor,
      valueCheckNumber = valueCheckNumber,
      generateValueCheck = generateValueCheck,
      setExecOrderAndSymbols = setExecOrderAndSymbols,
      generateDot = generateDot,
   }
   return self
end

return Debugger
