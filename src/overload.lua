local overloads = { }
local nodeApply = nil

local function module(name, table, fn)
   local mm = {
      name = name,
      table = table,
      functions = { },
      classes = { }
   }
   local overload = function(table, fnName, gradFn, capture)
      local old = table[fnName]
      if old ~= nil then
         local fnDesc = {
            name = name .. "." .. fnName,
            fn = old
         }
         local newFn = function(...)
            return nodeApply(fnDesc, gradFn, capture, ...)
         end
        return {
            name = fnName,
            newFn = newFn,
            oldFn = old
         }
      end
   end
   local overloadClass = function(table, className, fnName, gradFn, capture)
      local old = table[fnName]
      if old ~= nil then
         local fnDesc = {
            name = name .. "." .. className .. "." .. fnName,
            fn = old
         }
         local newFn = function(...)
            return nodeApply(fnDesc, gradFn, capture, ...)
         end
        return {
            name = fnName,
            newFn = newFn,
            oldFn = old
         }
      end
   end
   local overloadOp = function(table, opName, gradFn)
      local fnName = "__" .. opName
      local old = table[fnName]
      if old ~= nil then
         local fnDesc = {
            name = "op." .. fnName,
            operator = opName,
            fn = old
         }
         local newFn
         if opName == "unm" then
            newFn = function(a)
               return nodeApply(fnDesc, gradFn, true, a)
            end
         else
            newFn = function(a, b)
               return nodeApply(fnDesc, gradFn, true, a, b)
            end
         end
         return {
            name = fnName,
            newFn = newFn,
            oldFn = old
         }
      end
   end
   local moduleFns = {
      gradient = function(fnName, gradFn)
         local fn = overload(table, fnName, gradFn, true)
         mm.functions[#mm.functions + 1] = fn
      end,
      dynamic = function(...)
         local arg = {...}
         for i = 1, #arg do
            local fnName = arg[i]
            local fn = overload(table, fnName, nil, true)
            mm.functions[#mm.functions + 1] = fn
         end
      end,
      static = function(...)
         local arg = {...}
         for i = 1, #arg do
            local fnName = arg[i]
            local fn = overload(table, fnName, nil, false)
            mm.functions[#mm.functions + 1] = fn
         end
      end,
      operator = function(opName, gradFn)
         local fn = overloadOp(table, opName, gradFn)
         mm.functions[#mm.functions + 1] = fn
      end,
      class = function(className, fn)
         local classTable = table[className]
         local cc = {
            name = className,
            functions = { }
         }
         local classFns = {
            operator = function(opName, gradFn)
               local fn = overloadOp(classTable, opName, gradFn)
               cc.functions[#cc.functions + 1] = fn
            end,
            gradient = function(fnName, gradFn)
               local fn = overloadClass(classTable, className, fnName, gradFn, true)
               cc.functions[#cc.functions + 1] = fn
            end,
            dynamic = function(...)
               local arg = {...}
               for i = 1, #arg do
                  local fnName = arg[i]
                  local fn = overloadClass(classTable, className, fnName, nil, true)
                  cc.functions[#cc.functions + 1] = fn
               end
            end,
            static = function(...)
               local arg = {...}
               for i = 1, #arg do
                  local fnName = arg[i]
                  local fn = overload(classTable, fnName, nil, false)
                  cc.functions[#cc.functions + 1] = fn
               end
            end,
         }
         fn(classFns)
         mm.classes[#mm.classes + 1] = cc
      end
   }
   fn(moduleFns)
   overloads[#overloads + 1] = mm
end

-- Allow number * tensor style operations

function unwrapNumberValue(v)
   if type(v) == 'table' then
      return v.raw
   else
      return v
   end
end

local numberMetatable = {
   __add = function(a,b)
      if type(a) == "number" and torch.isTensor(b) then
         return b + a
      elseif type(a) == "number" and type(b) == "table" then
         return a + unwrapNumberValue(b)
      else
         return a + b
      end
   end,
   __sub = function(a,b)
      if type(a) == "number" and torch.isTensor(b) then
         return -b + a
      elseif type(a) == "number" and type(b) == "table" then
         return a - unwrapNumberValue(b)
      else
         return a - b
      end
   end,
   __mul = function(a,b)
      if type(a) == "number" and torch.isTensor(b) then
         return b * a
      elseif type(a) == "number" and type(b) == "table" then
         return a * unwrapNumberValue(b)
      else
         return a * b
      end
   end
}

local function install(fn)
   debug.setmetatable(1.0, numberMetatable)
   nodeApply = fn
   for i = 1, #overloads do
      local mm = overloads[i]
      for k = 1, #mm.functions do
         local fn = mm.functions[k]
         mm.table[fn.name] = fn.newFn
      end
      for k = 1, #mm.classes do
         local cc = mm.classes[k]
         local mt = torch.getmetatable('torch.' .. cc.name)
         for f = 1, #cc.functions do
            local fn = cc.functions[f]
            rawset(mt, fn.name, fn.newFn)
         end
      end
   end
end

local function uninstall()
   debug.setmetatable(1.0, nil)
   for i = 1, #overloads do
      local mm = overloads[i]
      for k = 1, #mm.functions do
         local fn = mm.functions[k]
         mm.table[fn.name] = fn.oldFn
      end
      for k = 1, #mm.classes do
         local cc = mm.classes[k]
         local mt = torch.getmetatable('torch.' .. cc.name)
         for f = 1, #cc.functions do
            local fn = cc.functions[f]
            rawset(mt, fn.name, fn.oldFn)
         end
      end
   end
end

-- Main functions:
local overload = {
   install = install,
   uninstall = uninstall,
   module = module
}

-- Return package
return overload

