local Source = require 'autograd.runtime.codegen.Source'
local Value = require 'autograd.runtime.codegen.Value'
local StringBuilder = require 'autograd.StringBuilder'
local stringx = require 'pl.stringx'

local function Debugger(opt)
   opt = opt or { }
   local debugHook = opt.debugHook

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
         local lines = stringx.split(tb, "\n")
         table.remove(lines, 1) -- Remove the header line
         local infos = { }
         for i,line in ipairs(lines) do
            local info = debug.getinfo(i)
            if info ~= nil then
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

   local main
   local function setMain(symbols, grads, answers)
      main = {
         symbols = symbols,
         grads = grads,
         answers = answers,
      }
   end

   local function setCode(code)
      main.code = code
   end

   local function walkGraph(value, node, parentNode, callback)
      callback(value, node, parentNode)
      if node then
         for i,input in ipairs(node.inputs) do
            walkGraph(input, input.source.node, node, callback)
         end
      end
   end

   local function valueKey(value)
      return 'x' .. value.source:symbolPath(main.symbols):gsub("[^a-zA-Z0-9]+", "_")
   end

   local function valueName(value)
      for _,grad in pairs(main.grads) do
         if grad.grad == value then
            return "grad(" .. grad.param.source:symbolPath(main.symbols) .. ")", "trapezium"
         end
      end
      for i,answer in ipairs(main.answers) do
         if answer == value then
            return "answer[" .. i .. "]", "octagon"
         end
      end
      local shape = "ellipse"
      if value.source.type ~= Source.COMPUTED then
         shape = "invtrapezium"
      end
      return value.source:symbolPath(main.symbols), shape
   end

   local function generateDotValue(out, value)
      local label, shape = valueName(value)
      local parts = { label }
      if torch.isTensor(value.raw) then
         table.insert(parts, torch.typename(value.raw):sub(7) .. "(" .. table.concat(value.raw:size():totable(), ", ") .. ")")
      elseif torch.isStorage(value.raw) then
         table.insert(parts, torch.typename(value.raw):sub(7) .. "(" .. value.raw:size() .. ")")
      end
      local color = "black"
      if value.debug and value.debug.min ~= nil then
         table.insert(parts, "[" .. value.debug.min .. ", " .. value.debug.max .. "]")
         if isNanOrInf(value.debug.min) or isNanOrInf(value.debug.max) then
            color = "red"
         end
      end
      out.write('\t' .. valueKey(value) .. ' [label="<' .. table.concat(parts, '<BR/>') .. '>" color="' .. color .. '" shape="' .. shape .. '"];\n')
   end

   local function generateDotNode(out, node)
      debugNode(node)
      local label = node.forwardFn.name
      color = "black"
      for _,output in ipairs(node.outputs) do
         if output.debug and (isNanOrInf(output.debug.min) or isNanOrInf(output.debug.max)) then
            color = "red"
            local forwardNode = rcsvFindCallStack(node)
            if forwardNode then
               for i,info in ipairs(forwardNode.debug.callStack) do
                  label = label .. "<BR/>" .. i .. ": " .. tostring(info.name) .. info.source .. ":" .. info.currentline
               end
            end
         end
      end
      out.write('\tnode' .. node.debug.index .. ' [label="<' .. label .. '>" color="'..color..'" shape="box"];\n')
   end

   local function generateEdge(out, node, value, reverse)
      local color = (node.debug.isForward and 'green') or 'blue'
      if reverse then
         out.write('\t' .. valueKey(value) .. ' -> node' .. node.debug.index .. ' [color="'..color..'"];\n')
      else
         out.write('\tnode' .. node.debug.index .. ' -> ' .. valueKey(value) .. ' [color="'..color..'"];\n')
      end
   end

   local function generateDot(fileName, value, node)
      local out = StringBuilder(fileName)
      out.write('digraph graphname {\n')
      local seen = { }
      local function callback(value, node, parentNode)
         if value and not seen[value] then
            seen[value] = true
            generateDotValue(out, value)
         end
         if node and not seen[node] then
            seen[node] = true
            generateDotNode(out, node)
         end
         if node and value then
            generateEdge(out, node, value)
         end
         if parentNode and value then
            generateEdge(out, parentNode, value, true)
         end
      end
      if value then
         -- Walk from the provided root value and node
         walkGraph(value, node, nil, callback)
      else
         -- Walk the entire graph
         for _,grad in ipairs(main.grads) do
            walkGraph(grad.grad, nil, nil, callback)
         end
         for _,answer in ipairs(main.answers) do
            walkGraph(answer, nil, nil, callback)
         end
      end
      out.write('}\n')
      return out.finish()
   end

   local function generateJson(fileName, value, node)
      local dot = generateDot(nil, value, node)
      local _,_,name,graph = dot:find('digraph%s*(%w*)%s*{(.*)}')
      local elts = stringx.split(stringx.strip(graph),'\n')

      local edges = {}
      local nodes = {}

      local function parseMeta(meta)
         local rest = meta
         local _,key,val
         local elts = {}
         while true do
            _,_,key,val,rest = rest:find('(.-)%=%"(.-)%"%s*(.*)')
            if not rest then break end
            elts[key] = val
         end
         return elts
      end

      for i,elt in ipairs(elts) do
         local elt = stringx.strip(elt)
         local _,_,content,meta = elt:find('(.-)%[(.*)%];$')
         meta = parseMeta(meta)
         if content:find('%-') then
            -- edge
            local _,_,name1,edge,name2 = content:find('^(.-) (.-) (.*)$')
            table.insert(edges, {
               from = stringx.strip(name1),
               to = stringx.strip(name2),
               edge = edge,
               meta = meta,
            })
         else
            -- node
            local name = stringx.strip(content)
            nodes[name] = {
               name = name,
               meta = meta,
            }
         end
      end

      local graph = {
         name = name,
         nodes = nodes,
         edges = edges,
      }

      local f = io.open(fileName, 'w')
      f:write(require('cjson').encode(graph))
      f:close()
   end

   local function showDot(value, node)
      if sys.uname() ~= 'macos' then
         print('showDot() only implemented on OSX')
         return
      end
      local fileName = os.tmpname()
      generateDot(fileName, value, node)
      os.execute('dot -O -Tsvg ' .. fileName)
      os.remove(fileName)
      os.execute('open -a Safari ' .. fileName ..'.svg')
   end

   local function valueCheck(value, raw, min, max, node)
      value.debug = value.debug or { }
      value.debug.min = (value.debug.min ~= nil and math.min(value.debug.min, min)) or min
      value.debug.max = (value.debug.max ~= nil and math.max(value.debug.max, max)) or max
      if isNanOrInf(value.debug.min) or isNanOrInf(value.debug.max) then
         local debugger = {
            generateDot = function(fileName) generateDot(fileName, value, node) end,
            generateJson = function(fileName) generateJson(fileName, value, node) end,
            showDot = function() showDot(value, node) end,
         }
         local msg = "autograd debugger detected a nan or inf value for " .. valueName(value)
         local forwardNode = rcsvFindCallStack(node)
         if forwardNode then
            for i,info in ipairs(forwardNode.debug.callStack) do
               msg = msg .. "\n\t\t" .. i .. ": " .. tostring(info.name) .. info.source .. ":" .. info.currentline
            end
         end
         local info = debug.getinfo(3)
         debugHook(debugger, msg, {
            source = info.source,
            line = info.currentline - 1
         })
      end
   end

   local function outputCheckTensor(nodeIndex, outputIndex, raw)
      local node = debugNodes[nodeIndex]
      local value = node.outputs[outputIndex]
      valueCheck(value, raw, raw:min(), raw:max(), node)
   end

   local function outputCheckNumber(nodeIndex, outputIndex, raw)
      local node = debugNodes[nodeIndex]
      local value = node.outputs[outputIndex]
      valueCheck(value, raw, raw, raw, node)
   end

   local function generateOutputCheck(node, outputIndex, symbol, out)
      debugNode(node)
      if node.forwardFn.operator == nil then
         local fnName = string.gsub(node.forwardFn.name, "%.", "_")
         if fnName:sub(#fnName - 3, #fnName) == "_new" then
            -- Don't check new memory as it contains random junk
            return
         end
      end
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

   return {
      captureCallStack = captureCallStack,
      setMain = setMain,
      setCode = setCode,
      generateDot = generateDot,
      showDot = showDot,
      outputCheckTensor = outputCheckTensor,
      outputCheckNumber = outputCheckNumber,
      generateOutputCheck = generateOutputCheck,
      inputCheckTensor = inputCheckTensor,
      inputCheckNumber = inputCheckNumber,
      generateInputCheck = generateInputCheck,
   }
end

return Debugger
