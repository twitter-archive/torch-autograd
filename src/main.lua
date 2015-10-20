-- TODO
-- Disallow overwriting anything
-- Tables

-- Deps
local haveCutorch,cutorch = pcall(require,'cutorch')
local debug = require 'debug'
local node = require 'autograd.node'
local overload = require 'autograd.overload'
local isTensor = require 'autograd.util'.isTensor
local Node = node.Node
local nodeApply = node.nodeApply
local getOutgrad = node.getOutgrad
local newStartNode = node.newStartNode
local isNode = node.isNode
local getValue = node.getValue
local debugFns = overload.debugFns

require 'pl'
require 'trepl'

-- For debugging
local function printSize(a)
   if type(a) == "number" then
      print("1x1")
   elseif isTensor(a) then
      print(torch.totable(a:size()))
   else
      print("???")
   end
end

-- Make sure we've got the right thing going on
local function checkInput(arg)
   if isTensor(arg) then
      local isValidType = false
      for _,tensorType in pairs(overload.tensorTypes) do
         isValidType = isValidType or 'torch.' .. tensorType == torch.typename(arg)
      end
      if not isValidType then
         local errMsg = "Input tensor is invalid type " .. torch.typename(arg) .. ". Valid types are"
         for _, tensorType in pairs(tensorTypes) do
            errMsg = errMsg .. " " .. tensorType
         end
         error(errMsg)
      end
   end
end

local lastTape = { }

-- Step through the computation graph and find the gradient
local function grad(fun, argnum, returnTape)
   argnum = argnum or 1
   local doGrad = function(...)
      local arg = {...}
      local tape = lastTape
      tape.nextIndex = 1

      -- Check the argument, to make sure it's alright.
      checkInput(arg[argnum])

      -- If our target argument is a table, we'll need to walk its members and node-ify them.
      -- For now, if we see a number or a tensor, we'll node-ify it, otherwise,
      -- if it's a table, we'll try to walk it
      arg[argnum] = newStartNode(arg[argnum], tape)
      local allAns = {fun(unpack(arg))}
      local ans = allAns[1]
      if not isNode(ans) then
         error("A node type was not returned. This is either because a gradient was not defined, or the input is independent of the output")
      end
      if type(getValue(ans)) ~= "number" then
         print("")
         print("Autograd only supports scalar outputs. This is current functions output: ")
         print(getValue(ans))
         error("Autograd only supports scalar return values. Output is not scalar")
      end

      ans.outgrad = 1.0

      overload.endRecording()
      for i=tape.nextIndex-1,1,-1 do
         local node = tape[i]
         if debugFns.preGradFn then
            debugFns.preGradFn(node)
         end
         for iarg=1,#node.args do
            local thisArg = node.args[iarg]
            if getmetatable(thisArg) == Node then
               if node.outgrad == nil then
                  if isTensor(node.value) then
                     node.outgrad = node.value.new(node.value:size()):zero()
                  elseif type(node.value) == "number" then
                     node.outgrad = 0.0
                  end
               end
               local gradUpdate = (node.gradFun[iarg+1])(node.outgrad, node.value, unpack(node.argValues))
               if thisArg.outgrad == nil or thisArg.outgrad == 0 then
                  thisArg.outgrad = gradUpdate
               else
                  thisArg.outgrad = thisArg.outgrad + gradUpdate
               end
            end
         end
         if debugFns.postGradFn then
            debugFns.postGradFn(node)
         end
      end
      overload.beginRecording()

      -- Now spit out the grads, along with any answers returned along the way
      local out = {}
      out[1] = getOutgrad(arg[argnum])

      local ansVal = getValue(allAns)
      if type(allAns) == "table" then
         for key,value in pairs(ansVal) do
            out[#out+1] = getValue(value)
         end
      else
         out[2] = ansVal
      end
      return unpack(out)
   end
   return doGrad
end

overload.install()
overload.beginRecording()

-- Main functions:
local autograd = {
   grad = grad,
   debugFns = debugFns,
   defineGradient = overload.defineGradient
}

-- Shortcut:
setmetatable(autograd, {
   __call = function(self,...)
      return grad(...)
   end
})

-- Return package
return autograd
