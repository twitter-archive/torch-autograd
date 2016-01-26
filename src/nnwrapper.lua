local util = require 'autograd.util'

local nodeApply
local function directApply(fun, gradFun, ...)
   return fun.fn(...)
end

local function setApplyFn(fn)
   nodeApply = fn or directApply
end
setApplyFn()

local function hasParams(nnObject)
   local hasParamFn, params = pcall(nnObject.parameters, nnObject)
   params = params or {}
   if not hasParamFn or #params == 0 then
      return false
   else
      return true
   end
end

local function isCriterion(nnObject)
   local isCriterion = false
   local mt = getmetatable(nnObject)
   if mt then
      local mmt = getmetatable(mt)
      if mmt then
         if mmt.__typename == 'nn.Criterion' then
            isCriterion = true
         end
      end
   end
   return isCriterion
end

local function isModule(nnObject)
   local isModule = false
   local mt = getmetatable(nnObject)
   if mt then
      local mmt = getmetatable(mt)
      if mmt then
         local t
         local mmmt = getmetatable(mmt)
         if mmmt then
            t = mmmt.__typename
         else
            t = mmt.__typename
         end
         if t == "nn.Module" or t == "nn.Sequential" or t == "nn.Container" or t == "nn.Threshold" then
            isModule = true
         end
      end
   end
   return isModule
end

local function getInputType(x)
   local dataType = nil
   if torch.isTensor(x) then
      dataType = torch.type(x)
   elseif type(x) == "table" then
      if x[1] then
         dataType = torch.type(x[1])
      end
   end
   return dataType
end

local function updateType(nnObject, lastType, newType)
   if not newType then
      error("Input is neither a tensor or a table of tensors. Type is " .. type(newType))
   end
   if lastType ~= newType then
      lastType = newType
      nnObject:type(newType)
   end
   return nnObject, lastType
end

local function wrapCriterion(nnObject)
   local lastType = ""
   local mod = {}

   local function forward(x, y)
      nnObject, lastType = updateType(nnObject, lastType, getInputType(x))
      return nnObject:forward(x, y)
   end

   local function backward(g, x, y)
      nnObject, lastType = updateType(nnObject, lastType, getInputType(x))
      return nnObject:backward(x, y)
   end

   local fn = function(x, y)
      local backFnDesc = {
         object = mod,
         method = "backward",
         name = "criterion",
         fn = backward,
         capture = true,
      }
      local gradFn = {
         function(g,ans,x,y)
            return nodeApply(backFnDesc, nil, g, x, y)
         end,
         function(g,ans,x,y)
            -- NOTE: should we throw error as uniplemented here?
            return util.fillSameSizeAs(y, 0)
         end,
      }
      local fnDesc = {
         object = mod,
         method = "forward",
         name = "criterion",
         fn = forward,
         capture = true,
      }
      return nodeApply(fnDesc, gradFn, x, y)
   end

   mod.entry = fn
   mod.forward = forward
   mod.backward = backward

   -- Shortcut:
   setmetatable(mod, {
      __call = function(self, ...)
         return self.entry(...)
      end
   })

   return mod
end

local function wrapModuleWithoutParams(nnObject)
   local lastType = ""
   local mod = {}
   local function forward(x)
      nnObject, lastType = updateType(nnObject, lastType, getInputType(x))
      return nnObject:forward(x)
   end

   local function backward(g,x)
      -- NOTE: Is this necessary if it's done forward?
      nnObject, lastType = updateType(nnObject, lastType, getInputType(x))
      nnObject:zeroGradParameters()
      local gradInput = nnObject:backward(x, g)
      return gradInput
   end

   local fn = function(x)
      local grads = nil
      local backFnDesc = {
         object = mod,
         method = "backward",
         name = "model",
         fn = backward,
         capture = true,
      }
      local gradFn = {
         function(g,ans,x)
            return nodeApply(backFnDesc, nil, g, x)
         end
      }
      local fnDesc = {
         object = mod,
         method = "forward",
         name = "model",
         fn = forward,
         capture = true,
      }
      return nodeApply(fnDesc, gradFn, x)
   end

   mod.entry = fn
   mod.forward = forward
   mod.backward = backward

   -- Shortcut:
   setmetatable(mod, {
      __call = function(self, ...)
         return self.entry(...)
      end
   })

   return mod
end

local function wrapModuleWithParams(nnObject)
   local lastType = ""
   local mod = {}
   local params = nnObject:parameters()
   local function forward(params,x)
      nnObject, lastType = updateType(nnObject, lastType, getInputType(x))

      local modelParams, modelGradParams = nnObject:parameters()
      for i,p in ipairs(modelParams) do
         if p ~= params[i] then
            -- NOTE: need a better error message
            -- if there's a type mismatch
            p:view(params[i], params[i]:size())
         end
      end
      return nnObject:forward(x)
   end

   local function backward(g,params,x)
      -- NOTE: Is this necessary if it's done forward?
      nnObject, lastType = updateType(nnObject, lastType, getInputType(x))
      local modelParams, modelGradParams = nnObject:parameters()
      for i,p in ipairs(modelParams) do
         if p ~= params[i] then
            p:view(params[i], params[i]:size())
         end
      end
      nnObject:zeroGradParameters()
      local gradInput = nnObject:backward(x, g)
      return {modelGradParams, gradInput}
   end

   local fn = function(params, x)
      local grads = nil
      local backFnDesc = {
         object = mod,
         method = "backward",
         name = "model",
         fn = backward,
         capture = true,
      }
      local gradFn = {
         function(g,ans,params,x)
            if grads == nil then
               grads = nodeApply(backFnDesc, nil, g, params, x)
            end
            return grads[1]
         end,
         function(g,ans,params,x)
            if grads == nil then
               grads = nodeApply(backFnDesc, nil, g, params, x)
            end
            return grads[2]
         end,
      }
      local fnDesc = {
         object = mod,
         method = "forward",
         name = "model",
         fn = forward,
         capture = true,
      }
      return nodeApply(fnDesc, gradFn, params, x)
   end

   mod.entry = fn
   mod.forward = forward
   mod.backward = backward

   -- Shortcut:
   setmetatable(mod, {
      __call = function(self, ...)
         return self.entry(...)
      end
   })

   return mod, params
end

-- Take in an nn module and functionalize it
local functionalize, functionalizePackage
functionalize = function(nnObject)
   if type(nnObject) == "string" then
      return functionalizePackage(nnObject)
   end

   if isModule(nnObject) then
      if hasParams(nnObject) then
         return wrapModuleWithParams(nnObject)
      else
         return wrapModuleWithoutParams(nnObject)
      end
   elseif isCriterion(nnObject) then
      return wrapCriterion(nnObject)
   else
      error("Input is not a package name or nn object")
   end
end

functionalizePackage = function(packageName)
   assert(type(packageName) == 'string')
   local loaded, mod = pcall(require, packageName)
   if not loaded then
      error("Could not load package '" .. packageName .. "'")
   else
      -- Iterate through every module in the package,
      -- and functionalize it
      local map = {}

      for modName, nnClass in pairs(mod) do
         if isModule(nnClass) or isCriterion(nnClass) then
            map[modName] = function(...)
               local out = {functionalize(nnClass(...))}
               return table.unpack(out)
            end
         end
      end
      return map
   end
end


return {
   functionalize = functionalize,
   functionalizePackage = functionalizePackage,
   setApplyFn = setApplyFn
}
