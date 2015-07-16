local keys = function(tbl)
    local keyset={}
    local n=0

    for k,v in pairs(tbl) do
      n=n+1
      keyset[n]=k
    end
    return keyset
end


function map(func, tbl)
    local newtbl = {}
    for i,v in pairs(tbl) do
        newtbl[i] = func(v)
    end
    return newtbl
end

function filter(func, tbl)
    local newtbl= {}
    for i,v in pairs(tbl) do
        if func(v) then
        newtbl[i]=v
        end
    end
    return newtbl
 end

function isnode(n)
    return class.type(n) == "Node"
end

function getval(v)
    if isnode(v) then
        return v.value
    else
        return v
    end
end
