local Debugger = require 'autograd.runtime.codegen.Debugger'
local Graph = require 'autograd.runtime.codegen.Graph'
local Value = require 'autograd.runtime.codegen.Value'
local LuaBackend = require 'autograd.runtime.codegen.backend.lua'

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

local function execUncached(fn, args, opt, nestedGradient)
   local graph = Graph.record(fn, args, opt)
   local retValues = { Value.collectGrads(graph.params[opt.argnum], graph.intermediateGrads) }
   for i = 1, #graph.answers do
      retValues[#retValues + 1] = graph.answers[i]
   end
   if not nestedGradient then
      retValues = Value.flatten(retValues)
   end
   return table.unpack(retValues)
end

local function printPoolStats(tensorPool)
   local size = 0
   for i = 1, #tensorPool do
      local tensor = tensorPool[i]
      size = size + tensor:storage():size() * 4
   end
   print("tensor pool size: " .. (size / (1024 * 1024)) .. " MB")
end

local function generateFn(fn, args, opt)
   if opt.debugHook then
      opt.debugger = Debugger(opt)
   end
   local graph = Graph.record(fn, args, opt)
   graph:optimize()
   return LuaBackend.generateFn(graph, opt)
end

local function copyStableTensors(retValues, stableGrads)
   for k, rv in pairs(retValues) do
      local sv = stableGrads[k]
      if sv == nil then
         sv = rv:clone()
         stableGrads[k] = sv
      end
      if type(rv) ~= type(sv) then
         error("mismatched types in stable tensor copy")
      end
      if torch.isTensor(rv) and rv ~= sv then
         if not torch.isSameSizeAs(rv, sv) then
            print("resizing stable grad " .. table.concat(sv:size():totable(), "x") .. " -> " .. table.concat(rv:size():totable(), "x"))
            sv:resize(rv:size())
         end
         sv:copy(rv)
         retValues[k] = sv
      elseif type(sv) == "table" then
         copyStableTensors(rv, sv)
      end
   end
end

local function create(fn, opt)
   local generatedFunctions = { }
   opt.tensorPool = { }
   opt.tensorLocals = { }
   local stableGradTensors = nil
   return function(...)
      local args = {...}
      if Graph.reentryDepth() > 0 then
         -- If we're in the middle of building the graph for a parent function, include this one in the parent, don't codegen.
         return execUncached(fn, args, opt, true)
      end
      local sigFun = opt.signatureFn or function(params)
         local tensorDims = { }
         buildSignature(params, tensorDims)
         return table.concat(tensorDims, "-")
      end
      local signature = sigFun(args)
      if signature == nil then
         return execUncached(fn, args, opt, false)
      end
      if generatedFunctions[signature] == nil then
         local gradFn, retValues, code = generateFn(fn, args, opt)
         --print(code)
         generatedFunctions[signature] = gradFn
         -- We already have the answers, don't run it all over again.
         if opt.withGradients and opt.withForward and not opt.debugHook then
            if opt.stableGradients then
               if stableGradTensors == nil then
                  stableGradTensors = retValues[1]
               else
                  -- Since the user is expecting the results in the same tensors, copy the new results to the first set of results.
                  copyStableTensors(retValues[1], stableGradTensors)
               end
            end
            return table.unpack(retValues)
         elseif opt.withForward and not opt.debugHook then
            return table.unpack(retValues)
         end
      end
      if opt.stableGradients then
         local retValues = {generatedFunctions[signature](table.unpack(args))}
         copyStableTensors(retValues[1], stableGradTensors)
         return table.unpack(retValues)
      else
         return generatedFunctions[signature](table.unpack(args))
      end
   end
end

return {
   create = create
}
