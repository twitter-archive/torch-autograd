-- Register all nn grads into autograd
local loaded = {}

-- Generic auto-wrapper of every function exposed in given
-- package + arbitary instantiated nn container/module:
local function functionalize(nodeApply)
   return function(input)
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
                        local nnObject = v(unpack(args))
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

                        local fn = function(x, W, b)
                           local backFnDesc = {
                              package = input,
                              ctor = modName,
                              object = nnObject,
                              method = "backward",
                              name = modName .. "_backward",
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
                              ctor = modName,
                              object = nnObject,
                              method = "forward",
                              name = modName .. "_forward",
                              args = args,
                              fn = forward
                           }
                           return nodeApply(fnDesc, gradFn, true, x, W, b)
                        end

                        local mod = {
                           entry = fn,
                           forward = forward,
                           backward = backward
                        }

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
                        local nnObject = v(unpack(args))
                        local lastType = ""

                        local function forward(x, W, b)
                           local dataType = (W or x):type()
                           if lastType ~= dataType then
                              lastType = dataType
                              nnObject:type(dataType)
                           end

                           nnObject.weight = W
                           nnObject.bias = b

                           return nnObject:forward(x)
                        end

                        local function backward(g, x, W, b)
                           local dataType = (W or x):type()
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

                        local fn = function(x, W, b)
                           local grads = nil
                           local backFnDesc = {
                              package = input,
                              ctor = modName,
                              object = nnObject,
                              method = "backward",
                              name = modName .. "_backward",
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
                              object = nnObject,
                              ctor = modName,
                              object = nnObject,
                              method = "forward",
                              name = modName .. "_forward",
                              args = args,
                              fn = forward
                           }
                           return nodeApply(fnDesc, gradFn, true, x, W, b)
                        end

                        local mod = {
                           entry = fn,
                           forward = forward,
                           backward = backward
                        }

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
         -- input is assumed to be a module:
         local mod = input
         local nnObject = input
         local params = nnObject:parameters()

         -- Construct object:
         local lastType = ""

         local function forward(params, x)
            local dataType = x:type()
            if lastType ~= dataType then
               lastType = dataType
               nnObject:type(dataType)
            end

            local modelParams = nnObject:parameters()
            for i,p in ipairs(modelParams) do
               if p ~= params[i] then
                  p:view(params[i], params[i]:size())
               end
            end

            return nnObject:forward(x)
         end

         local function backward(g, params, x)
            local dataType = x:type()
            if lastType ~= dataType then
               lastType = dataType
               nnObject:type(dataType)
            end

            local modelParams,modelGradParams = nnObject:parameters()
            for i,p in ipairs(modelParams) do
               if p ~= params[i] then
                  p:view(params[i], params[i]:size())
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
end

return functionalize
