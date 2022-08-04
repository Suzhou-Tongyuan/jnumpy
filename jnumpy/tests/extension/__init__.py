from jnumpy import init_jl, exec_julia, include_src

init_jl()
include_src("extension.jl", __file__)
exec_julia("extension.init()")
