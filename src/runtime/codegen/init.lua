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

function pctStr(n, tot)
   return tostring(math.floor((n / tot) * 100.0)) .. "%"
end

function padMin(s, min)
   if #s < min then
      return s .. string.rep(" ", min - #s)
   end
   return s
end


local function printProfile(stats)
   print(" ")
   --print(string.format("[autograd] calls: %i", stats.calls))
   print(string.format("[autograd] code cache hit rate: %i%%", math.floor((stats.cacheHits / stats.calls) * 100.0)))
   print(string.format("[autograd] generated code paths: %i", stats.cacheMisses))
   local averageCodegen = (stats.codegenTime / stats.cacheMisses) * 1000.0
   local averageExec = (stats.executionTime / stats.cacheHits) * 1000.0
   -- codegen always executes the code once
   local totalCodegen = stats.codegenTime - ((averageExec * stats.cacheMisses) / 1000.0)
   local totalAll = stats.codegenTime + stats.executionTime + stats.externalTime
   print(string.format("[autograd] code gen time: average=%.2fms total=%.2fs pct=%s", averageCodegen - averageExec, totalCodegen, pctStr(totalCodegen, totalAll))) 
   print(string.format("[autograd] exec time:     average=%.2fms total=%.2fs pct=%s", averageExec, stats.executionTime, pctStr(stats.executionTime, totalAll)))
   print(string.format("[autograd] external time: average=%.2fms total=%.2fs pct=%s", (stats.externalTime / stats.calls) * 1000.0, stats.externalTime, pctStr(stats.externalTime, totalAll)))
   print(" ")
end

local function create(fn, opt)
   local generatedFunctions = { }
   opt.tensorPool = { }
   opt.tensorLocals = { }
   local stableGradTensors = nil
   local stats = {
      cacheHits = 0,
      cacheMisses = 0,
      calls = 0,
      externalTime = 0,
      codegenTime = 0,
      executionTime = 0,
      prevTimestamp = nil
   }
   return function(...)
      local args = {...}
      if Graph.reentryDepth() > 0 then
         -- If we're in the middle of building the graph for a parent function, include this one in the parent, don't codegen.
         return execUncached(fn, args, opt, true)
      end
      stats.calls = stats.calls + 1
      if opt.profile == 'summary' and math.fmod(stats.calls, opt.profileReportFrequency) == 0 then
         printProfile(stats)
      end
      if stats.prevTimestamp ~= nil then
         stats.externalTime = stats.externalTime + (sys.clock() - stats.prevTimestamp)
      end
      local sigFun = opt.signatureFn or function(params)
         local tensorDims = { }
         buildSignature(params, tensorDims)
         return table.concat(tensorDims, "-")
      end
      local signature = sigFun(args)
      if signature == nil then
         stats.cacheMisses = stats.cacheMisses + 1
         stats.prevTimestamp = sys.clock()
         return execUncached(fn, args, opt, false)
      end
      if generatedFunctions[signature] == nil then
         local genStart = sys.clock()
         local gradFn, retValues, code = generateFn(fn, args, opt)
         stats.codegenTime = stats.codegenTime + (sys.clock() - genStart)
         generatedFunctions[signature] = gradFn
         stats.cacheMisses = stats.cacheMisses + 1
         stats.prevTimestamp = sys.clock()
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
      stats.cacheHits = stats.cacheHits + 1
      local execStart = sys.clock()
      local retValues = {generatedFunctions[signature](table.unpack(args))}
      stats.executionTime = stats.executionTime + (sys.clock() - execStart)
      if opt.stableGradients then
         copyStableTensors(retValues[1], stableGradTensors)
      end
      stats.prevTimestamp = sys.clock()
      return table.unpack(retValues)
   end
end

return {
   create = create
}
