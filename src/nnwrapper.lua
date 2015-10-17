-- Register all nn grads into autograd
local autograd = require 'autograd.main'
local node = require 'autograd.node'

local loaded = {}

-- Generic auto-wrapper of every function exposed in given
-- package
local function functionalize(pkg)
   -- return pre-loaded package:
   if loaded[pkg] then
      return loaded[pkg]
   end

   local mod = require(pkg)
   local map = { }

   for k,v in pairs(mod) do
      local mt = getmetatable(v)
      if mt then
         local mmt = getmetatable(mt)
         if mmt then
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
                  local n = node.nodeApply(forward, x, W, b)
                  if node.isNode(n) then
                     n.gradfun = {
                        "Linear",
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
                  end
                  return n
               end
            end
         end
      end
   end

   loaded[pkg] = map
   return map
end

return functionalize
