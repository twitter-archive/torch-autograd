local StringBuilder = { }
StringBuilder.__index = StringBuilder

function StringBuilder.new()
   local v = { }
   setmetatable(v, StringBuilder)
   v.strings = { }
   return v
end

function StringBuilder:write(...)
   local arg = {...}
   for i = 1, #arg do
      self.strings[#self.strings + 1] = arg[i]
   end
end

function StringBuilder:writeln(...)
   self:write(...)
   self:write("\n")
end

function StringBuilder:indent(n)
   for i = 1, n do
      self:write("    ")
   end
end

function StringBuilder:finish()
   return table.concat(self.strings, "")
end

return StringBuilder

