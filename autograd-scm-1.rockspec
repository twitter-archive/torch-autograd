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
      ['autograd.init'] = 'autograd.lua',
      ['autograd.util'] = 'util.lua',
   },
}
