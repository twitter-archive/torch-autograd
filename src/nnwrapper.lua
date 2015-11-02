-- Register all nn grads into autograd
local autograd = require 'autograd.main'
local node = require 'autograd.node'

local loaded = {}

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

      for k,v in pairs(mod) do
         local mt = getmetatable(v)
         if mt then
            local mmt = getmetatable(mt)
            if mmt then
               if mmt.__typename == 'nn.Criterion' then
                  map[k] = function(...)
                     -- Construct object:
                     local nnObject = v(...)
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
                        return nnObject:backward(x, y)
                     end

                     return function(x, W, b)
                        local gradFn = {
                           k,
                           function(g,ans,x,y)
                              return backward(g, x, y)
                           end,
                           function(g,ans,x,y)
                              print'ici'
                              return y.new(y:size()):zero()
                           end,
                        }
                        return node.nodeApply(forward, gradFn, x, W, b)
                     end
                  end
               else
                  map[k] = function(...)
                     -- Construct object:
                     local nnObject = v(...)
                     local lastType = ""

                     local function forward(x, W, b)
                        local dataType = x:type()
                        if lastType ~= dataType then
                           lastType = dataType
                           nnObject:type(dataType)
                        end

                        nnObject.weight = W
                        nnObject.bias = b

                        return nnObject:forward(x)
                     end

                     local function backward(g, x, W, b)
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

                     return function(x, W, b)
                        local grads = nil
                        local gradFn = {
                           k,
                           function(g,ans,x,W,b)
                              if grads == nil then
                                 grads = backward(g, x, W, b)
                              end
                              return grads[1]
                           end,
                           function(g,ans,x,W,b)
                              if grads == nil then
                                 grads = backward(g, x, W, b)
                              end
                              return grads[2]
                           end,
                           function(g,ans,x,W,b)
                              if grads == nil then
                                 grads = backward(g, x, W, b)
                              end
                              return grads[3]
                           end
                        }
                        return node.nodeApply(forward, gradFn, x, W, b)
                     end
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

      return function(params, x)
         local grads = nil
         local gradFn = {
            tostring(forward),
            function(g,ans,params,x)
               if grads == nil then
                  grads = backward(g, params, x)
               end
               return grads[1]
            end,
            function(g,ans,params,x)
               if grads == nil then
                  grads = backward(g, x, params, x)
               end
               return grads[2]
            end,
         }
         return node.nodeApply(forward, gradFn, params, x)
      end, params
   end
end

return functionalize
