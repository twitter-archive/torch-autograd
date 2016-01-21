local isTensor = torch.isTensor
local overload = require 'autograd.overload'
local DirectNode = require 'autograd.runtime.direct.DirectNode'

local DirectTape = { }
local assignmentMap = { }
local reverseAssignmentMap = { }
local currentTape = { }

-- A wrapper for a function
-- Anytime we try to apply a function to some arguments,
-- we'd like to make sure that if we're passing nodes in,
-- that we unpack the value in those nodes, apply the function
-- to the underlying value, and then wrap the value in a node
function nodeApply(fun, gradFun, ...)
   local arg = {...}
   local parent = nil
   local values = { }
   local ln = #arg
   for k = 1, ln do
      local v = arg[k]
      if getmetatable(v) == DirectNode then
         local alias = assignmentMap[v]
         if alias ~= nil then
            arg[k] = alias
            parent = alias
            values[#values + 1] = alias.value
         else
            parent = v
            values[#values + 1] = v.value
         end
      elseif type(v) == "table" then
         local tableValue = {}
         for j,element in pairs(v) do
            if getmetatable(element) == DirectNode then
               parent = element
               tableValue[j] = element.value
            else
               tableValue[j] = element
            end
         end
         values[#values + 1] = tableValue
      else
         values[#values + 1] = v
      end
   end
   if fun.capture and parent ~= nil then
      if fun.unsupported then
         error("function " .. fun.name .. " not currently supported by autograd")
      end
      local value = fun.fn(table.unpack(values))
      local node = nil
      local tape = currentTape
      local o = tape[tape.nextIndex]
      if o ~= nil then
         o.value = value
         o.fun = fun
         o.gradFun = gradFun
         o.args = arg
         o.outgrad = nil
         o.argValues = values
         tape.nextIndex = tape.nextIndex + 1
         if fun.name == "DirectNode.__internal_set" then
            local reverse = reverseAssignmentMap[arg[1]]
            if reverse ~= nil then
               assignmentMap[reverse] = o
            end
            reverseAssignmentMap[o] = arg[1]
            assignmentMap[arg[1]] = o
         end
         return o
      end
      local newNode = DirectNode:init(value, fun, gradFun, arg, values, tape)
      if fun.name == "DirectNode.__internal_set" then
         local reverse = reverseAssignmentMap[arg[1]]
         if reverse ~= nil then
            assignmentMap[reverse] = newNode
         end
         assignmentMap[arg[1]] = newNode
      end
      return newNode
   else
      local values = {fun.fn(table.unpack(values))}
      return table.unpack(values)
   end
end

function DirectTape.funOnly(fun, tape, argnum, ...)
   local arg = {...}
   local tape = tape or {}
   tape.nextIndex = 1

   -- If our target argument is a table, we'll need to walk its members and node-ify them.
   -- For now, if we see a number or a tensor, we'll node-ify it, otherwise,
   -- if it's a table, we'll try to walk it
   currentTape = tape
   arg[argnum] = DirectNode.newStartNode(arg[argnum], tape)

   overload.install(nodeApply)

   assignmentMap = { }
   reverseAssignmentMap = { }
   local allAns = {fun(table.unpack(arg))}

   overload.uninstall()

   local ans = allAns[1]
   if not DirectNode.isNode(ans) then
      error("A node type was not returned. This is either because a gradient was not defined, or the input is independent of the output")
   end
   -- Now spit out the grads, along with any answers returned along the way
   local out = {}

   local ansVal = DirectNode.getValue(allAns)
   if type(allAns) == "table" then
      for key,value in pairs(ansVal) do
         out[#out+1] = DirectNode.getValue(value)
      end
   else
      out[1] = ansVal
   end
   return arg, allAns, tape, table.unpack(out)
end

function DirectTape.gradOnly(tape, arg, argnum, allAns, gradOutput)
   local ans = allAns[argnum]
   ans.outgrad = gradOutput
   for i=tape.nextIndex-1,1,-1 do
      local node = tape[i]
      for iarg=#node.args,1,-1 do
         local thisArg = node.args[iarg]
         if getmetatable(thisArg) == DirectNode then
            if node.outgrad == nil then
               if isTensor(node.value) then
                  node.outgrad = node.value.new(node.value:size()):zero()
               elseif type(node.value) == "number" then
                  node.outgrad = 0.0
               end
            end
            local gf = (node.gradFun or {})[iarg]
            if gf ~= nil then
               local gradUpdate = (gf)(node.outgrad, node.value, table.unpack(node.argValues))
               if gradUpdate then
                  if thisArg.outgrad == nil or thisArg.outgrad == 0 then
                     thisArg.outgrad = gradUpdate
                  else
                     thisArg.outgrad = thisArg.outgrad + gradUpdate
                  end
               end
            elseif node.fun.differentiable then
               error("missing gradient for argument " .. tostring(iarg) .. " in function " .. node.fun.name)
            end
         -- Special-casing table-valued arguments that contain nodes
         -- right now, this is just torch.cat
         elseif type(thisArg) == "table" then
            local hasNode = false
            for k, v in pairs(thisArg) do
               if getmetatable(v) == DirectNode then
                  hasNode = true
                  break
               end
            end
            if hasNode then
               if node.outgrad == nil then
                  if isTensor(node.value) then
                     node.outgrad = node.value.new(node.value:size()):zero()
                  elseif type(node.value) == "number" then
                     node.outgrad = 0.0
                  end
               end
               local gradUpdate = (node.gradFun[iarg])(node.outgrad, node.value, table.unpack(node.argValues))
               local la = #thisArg
               for isubArg=1,la do
                  if gradUpdate[isubArg] then
                     local thisSubArg = thisArg[isubArg]
                     if getmetatable(thisSubArg) == DirectNode then
                        if thisSubArg.outgrad == nil or thisSubArg.outgrad == 0 then
                           thisSubArg.outgrad = gradUpdate[isubArg]
                        else
                           thisSubArg.outgrad = thisSubArg.outgrad + gradUpdate[isubArg]
                        end
                     end
                  end
               end
            end
         end
      end
   end
   -- Now spit out the grads
   local out = DirectNode.getOutgrad(arg[argnum])
   return out
end

local lastTape = { }

-- Step through the computation graph and find the gradient
function DirectTape.grad(fun, argnum, partialGrad, ...)
   local all  = {DirectTape.funOnly(fun, lastTape, argnum, ...)}
   local arg, allAns, tape, out = all[1], all[2], all[3], all[4]
   local ans = allAns[1]
   if partialGrad == nil and type(DirectNode.getValue(ans)) ~= "number" then
      print("")
      print("Autograd only supports scalar outputs. This is current functions output: ")
      print(DirectNode.getValue(ans))
      error("Autograd only supports scalar return values. Output is not scalar")
   end
   partialGrad = partialGrad or 1.0
   local go = DirectTape.gradOnly(tape, arg, argnum, allAns, partialGrad)
   local fout = {}
   fout[1] = go
   local ansVal = DirectNode.getValue(allAns)
   if type(allAns) == "table" then
      for key,value in pairs(ansVal) do
         fout[#fout+1] = DirectNode.getValue(value)
      end
   else
      fout[2] = ansVal
   end
   return table.unpack(fout)
end

return DirectTape
