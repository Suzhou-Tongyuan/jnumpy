from jnumpy import init_jl, exec_julia, include_src

init_jl()
include_src(__file__, "extension.jl")
exec_julia("extension.init()")
