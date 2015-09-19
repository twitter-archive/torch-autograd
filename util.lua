local keys = function(tbl)
    local keyset={}
    local n=0

    for k,v in pairs(tbl) do
      n=n+1
      keyset[n]=k
    end
    return keyset
end

function count(tbl)
    return #keys(tbl)
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
        newtbl[#newtbl+1]=v
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

function oneHot(labels, n)
    --[[
    Assume labels is a 1D tensor of contiguous class IDs, starting at 1.
    Turn it into a 2D tensor of size labels:size(1) x nUniqueLabels

    This is a pretty dumb function, assumes your labels are nice.
    ]]
    local n = n or torch.max(labels)
    local nLabels = labels:size(1)
    local out = torch.FloatTensor(nLabels, n):fill(0)
    for i=1,nLabels do
        out[i][labels[i]] = 1.0
    end
    return out
end