package = "autograd"
version = "scm-1"

source = {
   url = "",
   branch = "master",
}

description = {
   summary = "Automatic differentiation for Torch.",
   homepage = "",
   license = "MIT",
}

dependencies = {
   "torch >= 7.0",
   "trepl",
   "penlight",
   "totem",
   "moses",
}

build = {
   type = "builtin",
   modules = {
      ['autograd.init'] = 'src/init.lua',
      ['autograd.main'] = 'src/main.lua',
      ['autograd.util'] = 'src/util.lua',
      ['autograd.node'] = 'src/node.lua',
      ['autograd.model'] = 'src/model/init.lua',
      ['autograd.model.common'] = 'src/model/common.lua',
      ['autograd.model.NeuralNetwork'] = 'src/model/NeuralNetwork.lua',
      ['autograd.model.SpatialNetwork'] = 'src/model/SpatialNetwork.lua',
      ['autograd.model.RecurrentNetwork'] = 'src/model/RecurrentNetwork.lua',
      ['autograd.model.RecurrentLSTMNetwork'] = 'src/model/RecurrentLSTMNetwork.lua',
      ['autograd.gradfuns'] = 'src/gradfuns.lua',
      ['autograd.overload'] = 'src/overload.lua',
      ['autograd.loss'] = 'src/loss/init.lua',
      ['autograd.test'] = 'test/test.lua',
      ['autograd.gradcheck'] = 'src/gradcheck.lua',
      ['autograd.nnwrapper'] = 'src/nnwrapper.lua',
   },
}
