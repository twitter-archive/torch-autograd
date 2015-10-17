-- Register all nnfunc grads into autograd
local autograd = require 'autograd.main'
local node = require 'autograd.node'

local loaded = {}

-- Generic auto-wrapper of every function exposed
-- by nnfunc:
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

--[[
-- What? This file provides a functionalize utility that
-- turns every nn Module into a simple function.

-- Package
local nnfunc = {}

-- Grads lookup
nnfunc.gradsOf = {}
local gradsOf = nnfunc.gradsOf

-- Functionalize any nn-like package:
function nnfunc.functionalize(mod)
   -- mod is the package name:
   assert(type(mod) == 'string', 'mod should be a string (e.g. nn, ...')

   -- populate hash of methods:
   nnfunc[mod] = {}
   local map = nnfunc[mod]

   -- lookup every module in source package:
   mod = require(mod)
   for k,v in pairs(mod) do
      local mt = getmetatable(v)
      if mt then
         local mmt = getmetatable(mt)
         if mmt then
            map[k] = function(...)
               -- Construct object:
               local o = v(...)
               local lastType = ""

               -- Gradients:
               local g = function(data)


               end

               -- Fprop:
               local f = function(data)
                  if data.gradOutput then
                     -- compute gradients
                     return g(data)
                  else

                  end
               end

               -- Register:
               gradsOf[f] = g

               -- Return both:
               return f,g
            end
         end
      end
   end
end

-- Functinoalize nn by default:
nnfunc.functionalize 'nn'

-- Tests
nnfunc.test = function()
   require('./test')
end

-- Return package:
return nnfunc

--]]

return functionalize
