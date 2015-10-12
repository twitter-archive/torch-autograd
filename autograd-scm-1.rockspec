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
   "class",
   "trepl",
   "moses",
   "penlight",
   "nnfunc",
   "totem",
}

build = {
   type = "builtin",
   modules = {
      ['autograd.init'] = 'src/init.lua',
      ['autograd.main'] = 'src/main.lua',
      ['autograd.util'] = 'src/util.lua',
      ['autograd.node'] = 'src/node.lua',
      ['autograd.number'] = 'src/number.lua',
      ['autograd.model'] = 'src/model.lua',
      ['autograd.loss'] = 'src/loss.lua',
      ['autograd.test'] = 'test/test.lua',
      ['autograd.gradcheck'] = 'src/gradcheck.lua',
      ['autograd.nnfuncwrapper'] = 'src/nnfuncwrapper.lua',
   },
}
