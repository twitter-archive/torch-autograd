local util = require 'autograd.util'

local Profiler = { }
Profiler.__index = Profiler

function Profiler.new()
   local p = { }
   p.lineMap = { }
   p.entries = { }
   p.times = 0
   setmetatable(p, Profiler)
   return p
end

function Profiler:mark(fun, level)
   local name = fun.name
   if fun.raw then
      name = fun.raw.__typename
      if name == nil or name == "" then
         name = "(nn object)"
      end
   end
   local di = debug.getinfo(level + 1)
   local line = di.short_src .. ":" .. di.currentline
   local fnMap = self.lineMap[line]
   if fnMap == nil then
      fnMap = { }
      self.lineMap[line] = fnMap
   end
   local entryIndex = fnMap[name]
   if entryIndex == nil then
      entryIndex = #self.entries + 1
      self.entries[entryIndex] = {
         debuginfo = di,
         name = name,
         line = line,
         forwardTime = 0,
         backwardTime = 0
      }
      fnMap[name] = entryIndex
   end
   return entryIndex
end

function Profiler:markCycle()
   self.times = self.times + 1
end

function Profiler:measureForward(id, time)
   self.entries[id].forwardTime = self.entries[id].forwardTime + time
end

function Profiler:measureBackward(id, time)
   self.entries[id].backwardTime = self.entries[id].backwardTime + time
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

function Profiler:printReport(type)
   local totalForward = 0
   local totalBackward = 0
   for i = 1, #self.entries do
      local t = self.entries[i]
      totalForward = totalForward + t.forwardTime
      totalBackward = totalBackward + t.backwardTime
   end

   local timeSorted = util.shallowCopy(self.entries)
   table.sort(timeSorted, function(a, b)
      return (a.forwardTime + a.backwardTime) > (b.forwardTime + b.backwardTime)
   end)
   print("")
   print(string.format("[autograd] average forward time: %.2fms", (totalForward / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average backward time: %.2fms", (totalBackward / (self.times + 1)) * 1000.0))
   print(string.format("[autograd] average overall time: %.2fms", ((totalForward + totalBackward) / (self.times + 1)) * 1000.0))
   print("[autograd] top operations:")
   if type == "detailed" then
      print("[autograd] " .. string.rep("=", 80))
      print("[autograd] " .. padMin("name", 20), "fwd", "bwd", "ovr", "line")
      print("[autograd] " .. string.rep("=", 80))
      for i = 1, math.min(10, #timeSorted) do
         local t = timeSorted[i]
         print("[autograd] " .. padMin(t.name, 20), pctStr(t.forwardTime, totalForward), pctStr(t.backwardTime, totalBackward), pctStr(t.forwardTime + t.backwardTime, totalForward + totalBackward), t.line)
      end
   end
   print("")
end

return Profiler
