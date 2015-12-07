-- Register all nn grads into autograd
local loaded = {}
local nodeApply

local function directApply(fun, gradFun, capture, ...)
   return fun.fn(...)
end

local function setApplyFn(fn)
   nodeApply = fn or directApply
end

setApplyFn()

-- Generic auto-wrapper of every function exposed in given
-- package + arbitary instantiated nn container/module:
local function functionalize(input)
   -- return pre-loaded package:
   if loaded[input] then
      return loaded[input]
   end

   -- input can be a pkg name or a module
   if type(input) == 'string' then
      -- input is a pkg name:
      local pkg = input
      local mod = require(pkg)
      local map = { }

      for modName, v in pairs(mod) do
         local mt = getmetatable(v)
         if mt then
            local mmt = getmetatable(mt)
            if mmt then
               if mmt.__typename == 'nn.Criterion' then
                  map[modName] = function(...)
                     local args = {...}
                     -- Construct object:
                     local nnObject = v(table.unpack(args))
                     local lastType = ""

                     local function forward(x, y)
                        local dataType = x:type()
                        if lastType ~= dataType then
                           lastType = dataType
                           nnObject:type(dataType)
                        end
                        return nnObject:forward(x, y)
                     end

                     local function backward(g, x, y)
                        local dataType = x:type()
                        if lastType ~= dataType then
                           lastType = dataType
                           nnObject:type(dataType)
                        end
                        return nnObject:backward(x, y)
                     end

                     local mod = { }

                     local fn = function(x, W, b)
                        local backFnDesc = {
                           object = mod,
                           method = "backward",
                           name = modName,
                           args = args,
                           fn = backward
                        }
                        local gradFn = {
                           function(g,ans,x,y)
                              return nodeApply(backFnDesc, nil, true, g, x, y)
                           end,
                           function(g,ans,x,y)
                              return util.fillSameSizeAs(y, 0)
                           end,
                        }
                        local fnDesc = {
                           package = input,
                           object = mod,
                           method = "forward",
                           name = modName,
                           args = args,
                           fn = forward
                        }
                        return nodeApply(fnDesc, gradFn, true, x, W, b)
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


               else
                  map[modName] = function(...)
                     -- Construct object:
                     local args = {...}
                     local nnObject = v(table.unpack(args))
                     local lastType = ""

                     local function forward(x, W, b)
                        local dataType
                        if torch.isTensor(x) then
                           dataType = (W or x):type()
                        elseif type(x) == "table" then
                           if x[1] then
                              dataType = (W or x[1]):type()
                           else
                              error("X is neither a Tensor, nor a table array")
                           end
                        end

                        if lastType ~= dataType then
                           lastType = dataType
                           nnObject:type(dataType)
                        end

                        nnObject.weight = W
                        nnObject.bias = b

                        return nnObject:forward(x)
                     end

                     local function backward(g, x, W, b)
                        local dataType
                        if torch.isTensor(x) then
                           dataType = (W or x):type()
                        elseif type(x) == "table" then
                           if x[1] then
                              dataType = (W or x[1]):type()
                           else
                              error("X is neither a Tensor, nor a table array")
                           end
                        end

                        if lastType ~= dataType then
                           lastType = dataType
                           nnObject:type(dataType)
                        end

                        nnObject.weight = W
                        nnObject.bias = b

                        if nnObject.gradWeight then
                           nnObject.gradWeight:zero()
                        end
                        if nnObject.gradBias then
                           nnObject.gradBias:zero()
                        end

                        local gradInput = nnObject:backward(x, g)

                        return {
                           gradInput,
                           nnObject.gradWeight,
                           nnObject.gradBias,
                        }
                     end

                     local mod = {}

                     local fn = function(x, W, b)
                        local grads = nil
                        local backFnDesc = {
                           object = mod,
                           method = "backward",
                           name = modName,
                           args = args,
                           fn = backward
                        }
                        local gradFn = {
                           function(g,ans,x,W,b)
                              if grads == nil then
                                 grads = nodeApply(backFnDesc, nil, true, g, x, W, b)
                              end
                              return grads[1]
                           end,
                           function(g,ans,x,W,b)
                              if grads == nil then
                                 grads = nodeApply(backFnDesc, nil, true, g, x, W, b)
                              end
                              return grads[2]
                           end,
                           function(g,ans,x,W,b)
                              if grads == nil then
                                 grads = nodeApply(backFnDesc, nil, true, g, x, W, b)
                              end
                              return grads[3]
                           end
                        }
                        local fnDesc = {
                           package = input,
                           object = mod,
                           method = "forward",
                           name = modName,
                           args = args,
                           fn = forward
                        }
                        return nodeApply(fnDesc, gradFn, true, x, W, b)
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
               end
            end
         end
      end

      loaded[pkg] = map
      return map

   else
      -- input is assumed to be an instantiated module
      local nnObject = input
      local hasParamFn, params = pcall(nnObject.parameters, nnObject)
      if not hasParamFn or #params == 0 then 
         params = {}
         hasParamFn = false
      end

      -- Construct object:
      local lastType = ""

      local function forward(...) -- {params, x} usually. If no params, then {x}
         local args = {...}
         local params, x
         if not hasParamFn then
            x = args[1]
            params = {}
         else
            params = args[1]
            x = args[2]
         end

         local dataType = (params[1] or x):type()
         if lastType ~= dataType then
            lastType = dataType
            nnObject:type(dataType)
         end

         if hasParamFn then
            modelParams = nnObject:parameters()
            for i,p in ipairs(modelParams) do
               if p ~= params[i] then
                  p:view(params[i], params[i]:size())
               end
            end
         end

         return nnObject:forward(x)
      end

      local function backward(...) -- {g, params, x} usually. If no params, then {g,x}

         local args = {...}
         local g, params, x
         if not hasParamFn then
            g = args[1]
            x = args[2]
            params = {}
         else
            g = args[1]
            params = args[2]
            x = args[3]
         end

         local dataType = (params[1] or x):type()
         if lastType ~= dataType then
            lastType = dataType
            nnObject:type(dataType)
         end

         if hasParamFn then
            modelParams, modelGradParams = nnObject:parameters()
            for i,p in ipairs(modelParams) do
               if p ~= params[i] then
                  p:view(params[i], params[i]:size())
               end
            end
         end

         nnObject:zeroGradParameters()

         local gradInput = nnObject:backward(x, g)

         return {
            modelGradParams,
            gradInput,
         }
      end

      local mod = {}

      local fn = function(params, x)
         local grads = nil
         local backFnDesc = {
            object = mod,
            method = "backward",
            name = "model",
            fn = backward
         }
         local gradFn = {
            function(g,ans,params,x)
               if grads == nil then
                  grads = nodeApply(backFnDesc, nil, true, g, params, x)
               end
               return grads[1]
            end,
            function(g,ans,params,x)
               if grads == nil then
                  grads = nodeApply(backFnDesc, nil, true, g, params, x)
               end
               return grads[2]
            end,
         }
         local fnDesc = {
            object = mod,
            method = "forward",
            name = "model",
            fn = forward
         }
         return nodeApply(fnDesc, gradFn, true, params, x)
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
end

return {
   functionalize = functionalize,
   setApplyFn = setApplyFn
}
