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

   local debugValues = { }
   local function debugValue(value)
      if not value.debug then
         table.insert(debugValues, value)
         value.debug = { index = #debugValues }
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
            if info.name == 'createGraph' then
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

   local function valueCheck(value, raw, min, max)
      value.debug = value.debug or { }
      value.debug.min = (value.debug.min ~= nil and math.min(value.debug.min, min)) or min
      value.debug.max = (value.debug.max ~= nil and math.max(value.debug.max, max)) or max
      if isNanOrInf(value.debug.min) or isNanOrInf(value.debug.max) then
         debugHook(self, "Detected NaN or Inf")
      end
   end

   local function outputCheckTensor(nodeIndex, outputIndex, raw)
      local node = debugNodes[nodeIndex]
      local value = node.outputs[outputIndex]
      valueCheck(value, raw, raw:min(), raw:max())
   end

   local function outputCheckNumber(nodeIndex, outputIndex, raw)
      local node = debugNodes[nodeIndex]
      local value = node.outputs[outputIndex]
      valueCheck(value, raw, raw, raw)
   end

   local function generateOutputCheck(node, outputIndex, symbol, out)
      debugNode(node)
      local output = node.outputs[outputIndex]
      if output.type == Value.TENSOR then
         out.write("    debugger.outputCheckTensor(" .. table.concat({ node.debug.index, outputIndex, symbol }, ", ") .. ")\n")
      elseif output.type == Value.NUMBER then
         out.write("    debugger.outputCheckNumber(" .. table.concat({ node.debug.index, outputIndex, symbol }, ", ") .. ")\n")
      end
   end

   local function inputCheckTensor(valueIndex, raw)
      local value = debugValues[valueIndex]
      valueCheck(value, raw, raw:min(), raw:max())
   end

   local function inputCheckNumber(valueIndex, raw)
      local value = debugValues[valueIndex]
      valueCheck(value, raw, raw, raw)
   end

   local function generateInputCheck(value, symbol, out)
      debugValue(value)
      if value.type == Value.TENSOR then
         out.write("    debugger.inputCheckTensor(" .. table.concat({ value.debug.index, symbol }, ", ") .. ")\n")
      elseif value.type == Value.NUMBER then
         out.write("    debugger.inputCheckNumber(" .. table.concat({ value.debug.index, symbol }, ", ") .. ")\n")
      elseif value.type == Value.TABLE then
         for k,v in pairs(value.raw) do
            generateInputCheck(v, symbol .. "." .. k, out)
         end
      end
   end

   local main
   local function setMain(execOrder, symbols, grads, answers)
      main = {
         execOrder = execOrder,
         symbols = symbols,
         grads = grads,
         answers = answers,
      }
   end

   local function generateDot(fileName)
      -- Figure out all the variables both supplied and intermediate
      local vars = { }
      local function varKey(io)
         return 'x' .. io.source:symbolPath(main.symbols):gsub("[^a-zA-Z0-9]+", "_")
      end
      local function addVar(io)
         local key = varKey(io)
         if not vars[key] then
            local parts = { }
            local label
            local shape = "ellipse"
            for _,grad in pairs(main.grads) do
               if grad.grad == io then
                  label = "grad(" .. grad.param.source:symbolPath(main.symbols) .. ")"
                  shape = "trapezium"
                  break
               end
            end
            for i,answer in ipairs(main.answers) do
               if answer == io then
                  label = "answer[" .. i .. "]"
                  shape = "octagon"
                  break
               end
            end
            if label then
               table.insert(parts, label)
            elseif io.source.type ~= Source.COMPUTED then
               table.insert(parts, io.source:symbolPath(main.symbols))
               shape = "invtrapezium"
            end
            if torch.isTensor(io.raw) then
               table.insert(parts, torch.typename(io.raw):sub(7) .. "(" .. table.concat(io.raw:size():totable(), ", ") .. ")")
            elseif torch.isStorage(io.raw) then
               table.insert(parts, torch.typename(io.raw):sub(7) .. "(" .. io.raw:size() .. ")")
            end
            local color = "black"
            if io.debug and io.debug.min ~= nil then
               table.insert(parts, "[" .. io.debug.min .. ", " .. io.debug.max .. "]")
               if isNanOrInf(io.debug.min) or isNanOrInf(io.debug.max) then
                  color = "red"
               end
            end
            vars[key] = { label = table.concat(parts, '<BR/>'), color = color, shape = shape }
         end
      end
      for i,node in ipairs(main.execOrder) do
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
         out.write('\t' .. key .. ' [label=<' .. var.label .. '> color="' .. var.color .. '" shape="' .. var.shape .. '"];\n')
      end
      local sawNan = false
      for i,node in ipairs(main.execOrder) do
         local label = node.forwardFn.name
         color = "black"
         for _,output in ipairs(node.outputs) do
            if output.debug and (isNanOrInf(output.debug.min) or isNanOrInf(output.debug.max)) then
               color = "red"
               if not sawNan then
                  local forwardNode = rcsvFindCallStack(node)
                  if forwardNode then
                     for i,info in ipairs(forwardNode.debug.callStack) do
                        label = label .. "<BR/>" .. i .. ": " .. tostring(info.name) .. info.source .. ":" .. info.currentline
                     end
                     sawNan = true
                  end
               end
            end
         end
         out.write('\tnode' .. i .. ' [label=<' .. label .. '> color="'..color..'" shape=box];\n')
      end
      for i,node in ipairs(main.execOrder) do
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

   local function showDot()
      if sys.uname() ~= 'macos' then
         print('showDow() only implemented on OSX')
         return
      end
      local file = os.tmpname()
      generateDot(file)
      os.execute('dot -O -Tsvg ' .. file)
      os.remove(file)
      os.execute('open -a Safari ' .. file..'.svg')
      -- TODO: gc that leaked svg
   end

   self = {
      captureCallStack = captureCallStack,
      outputCheckTensor = outputCheckTensor,
      outputCheckNumber = outputCheckNumber,
      generateOutputCheck = generateOutputCheck,
      inputCheckTensor = inputCheckTensor,
      inputCheckNumber = inputCheckNumber,
      generateInputCheck = generateInputCheck,
      setMain = setMain,
      generateDot = generateDot,
      showDot = showDot,
   }
   return self
end

return Debugger
