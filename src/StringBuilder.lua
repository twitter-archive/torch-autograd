
local function StringBuilder(fileName)
   local strs = { }
   local f
   if fileName then
      f = io.open(fileName, "w")
   end
   return {
      write = function(...)
         local arg = {...}
         for i = 1, #arg do
            if f then
               f:write(arg[i])
            else
               strs[#strs + 1] = arg[i]
            end
         end
      end,
      finish = function()
         if f then
            f:close()
         else
            return table.concat(strs, "")
         end
      end
   }
end

return StringBuilder
