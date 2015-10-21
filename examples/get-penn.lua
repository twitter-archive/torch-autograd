-- Dictionary
local word2id = {}
local id2word = {}

-- Load txt file:
local function loadDataset(path)
   -- Parse words:
   local data = io.open(path):read('*all')
   data = data:gsub('\n','<eos>')
   local tokens = stringx.split(data)

   -- Build dictionary:
   local id = 1
   local ids = torch.FloatTensor(#tokens)
   for i,token in ipairs(tokens) do
      if not word2id[token] then
         word2id[token] = id
         id2word[id] = token
         id = id + 1
      end
      ids[i] = word2id[token]
   end

   -- Final dataset:
   return ids
end

-- Get/create dataset
local function setupData()
   -- Fetch from Amazon
   if not path.exists(sys.fpath()..'/penn') then
      os.execute[[
      curl https://s3.amazonaws.com/torch.data/penn.tgz -o penn.tgz
      tar xvf penn.tgz
      rm penn.tgz
      ]]
   end

   -- Each dataset is a 1D tensor of ids, the 4th arg
   -- is the dictionary, with 2-way indexes
   return
      loadDataset(sys.fpath()..'/penn/train.txt'),
      loadDataset(sys.fpath()..'/penn/valid.txt'),
      loadDataset(sys.fpath()..'/penn/test.txt'),
      {word2id = word2id, id2word = id2word}
end

return setupData
