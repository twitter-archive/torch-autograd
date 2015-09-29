local ops = {
   add = function(a,b) return a+b end,
   sub = function(a,b) return a-b end,
   mul = function(a,b) return a*b end,
   div = function(a,b) return a/b end,
   pow = function(a,b) return a^b end,
   unm = function(a) return -1*a end -- TODO: more efficient way across numbers and torch?
}

return ops