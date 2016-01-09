local overloads = { }
local nodeApply = nil
local nnwrapper = require 'autograd.nnwrapper'

local toRegister = { }

local function module(name, table, fn)
   toRegister[#toRegister + 1] = function()
      local mm = {
         name = name,
         table = table,
         functions = { },
         classes = { }
      }
      local supported = { }
      local overload = function(table, fnName, gradFn, capture, differentiable, unsupported)
         local old = table[fnName]
         if old ~= nil then
            local fnDesc = {
               name = name .. "." .. fnName,
               differentiable = differentiable,
               fn = old,
               capture = capture,
               unsupported = unsupported,
            }
            local newFn = function(...)
               return nodeApply(fnDesc, gradFn, ...)
            end
            return {
               name = fnName,
               newFn = newFn,
               oldFn = old
            }
         end
      end
      local overloadClass = function(table, className, fnName, gradFn, capture, differentiable, unsupported)
         local old = table[fnName]
         if old ~= nil then
            local fnDesc = {
               name = name .. "." .. className .. "." .. fnName,
               differentiable = differentiable,
               fn = old,
               capture = capture,
               unsupported = unsupported,
            }
            local newFn = function(...)
               return nodeApply(fnDesc, gradFn, ...)
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
               differentiable = true,
               capture = true,
               fn = old
            }
            local newFn
            if opName == "unm" then
               newFn = function(a)
                  return nodeApply(fnDesc, gradFn, a)
               end
            else
               newFn = function(a, b)
                  return nodeApply(fnDesc, gradFn, a, b)
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
            local fn = overload(table, fnName, gradFn, true, true, false)
            supported[fnName] = true
            mm.functions[#mm.functions + 1] = fn
         end,
         dynamic = function(...)
            local arg = {...}
            for i = 1, #arg do
               local fnName = arg[i]
               local fn = overload(table, fnName, nil, true, true, false)
               supported[fnName] = true
               mm.functions[#mm.functions + 1] = fn
            end
         end,
         initializer = function(...)
            local arg = {...}
            for i = 1, #arg do
               local fnName = arg[i]
               local fn = overload(table, fnName, nil, true, false, false)
               supported[fnName] = true
               mm.functions[#mm.functions + 1] = fn
            end
         end,
         static = function(...)
            local arg = {...}
            for i = 1, #arg do
               local fnName = arg[i]
               local fn = overload(table, fnName, nil, false, false, false)
               supported[fnName] = true
               mm.functions[#mm.functions + 1] = fn
            end
         end,
         unsupported = function(...)
            local arg = {...}
            for i = 1, #arg do
               local fnName = arg[i]
               local fn = overload(table, fnName, nil, true, false, true)
               supported[fnName] = true
               mm.functions[#mm.functions + 1] = fn
            end
         end,
         ignore = function(...)
            local arg = {...}
            for i = 1, #arg do
               local fnName = arg[i]
               supported[fnName] = true
            end
         end,
         operator = function(opName, gradFn)
            local fn = overloadOp(table, opName, gradFn)
            supported[opName] = true
            mm.functions[#mm.functions + 1] = fn
         end,
         defaultUnsupported = function()
            for k, v in pairs(table) do
               if supported[k] == nil and type(v) == "function" and string.sub(k, 1, 2) ~= "__" then
                  local fn = overload(table, k, nil, true, false, true)
                  mm.functions[#mm.functions + 1] = fn
               end
            end
         end,
         class = function(className, fn)
            local classTable = table[className]
            local cc = {
               name = className,
               functions = { }
            }
            local supported = { }
            local classFns = {
               operator = function(opName, gradFn)
                  local fn = overloadOp(classTable, opName, gradFn)
                  supported[opName] = true
                  cc.functions[#cc.functions + 1] = fn
               end,
               gradient = function(fnName, gradFn)
                  local fn = overloadClass(classTable, className, fnName, gradFn, true, true, false)
                  supported[fnName] = true
                  cc.functions[#cc.functions + 1] = fn
               end,
               dynamic = function(...)
                  local arg = {...}
                  for i = 1, #arg do
                     local fnName = arg[i]
                     local fn = overloadClass(classTable, className, fnName, nil, true, true, false)
                     supported[fnName] = true
                     cc.functions[#cc.functions + 1] = fn
                  end
               end,
               initializer = function(...)
                  local arg = {...}
                  for i = 1, #arg do
                     local fnName = arg[i]
                     local fn = overloadClass(classTable, className, fnName, nil, true, false, false)
                     supported[fnName] = true
                     cc.functions[#cc.functions + 1] = fn
                  end
               end,
               static = function(...)
                  local arg = {...}
                  for i = 1, #arg do
                     local fnName = arg[i]
                     local fn = overloadClass(classTable, className, fnName, nil, false, false, false)
                     supported[fnName] = true
                     cc.functions[#cc.functions + 1] = fn
                  end
               end,
               unsupported = function(...)
                  local arg = {...}
                  for i = 1, #arg do
                     local fnName = arg[i]
                     local fn = overloadClass(classTable, className, fnName, nil, true, false, true)
                     supported[fnName] = true
                     cc.functions[#cc.functions + 1] = fn
                  end
               end,
               defaultUnsupported = function()
                  local mt = getmetatable(classTable)
                  for k, v in pairs(mt) do
                     if supported[k] == nil and type(v) == "function" and string.sub(k, 1, 2) ~= "__" then
                        local fn = overloadClass(classTable, className, k, nil, true, false, true)
                        cc.functions[#cc.functions + 1] = fn
                     end
                  end
               end
            }
            fn(classFns)
            mm.classes[#mm.classes + 1] = cc
         end
      }
      fn(moduleFns)
      overloads[#overloads + 1] = mm
   end
end

local installDepth = 0

local function install(fn)
   installDepth = installDepth + 1
   if installDepth ~= 1 then
      return
   end
   if #toRegister > 0 then
      for i = 1, #toRegister do
         toRegister[i]()
      end
      toRegister = { }
   end
   nnwrapper.setApplyFn(fn)
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
   installDepth = installDepth - 1
   if installDepth ~= 0 then
      return
   end
   nnwrapper.setApplyFn(nil)
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

