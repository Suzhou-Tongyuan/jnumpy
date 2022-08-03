from jnumpy import init_jl, exec_julia, include_src

init_jl()
include_src(__file__, "jl_example.jl") # use a relative path for "jl_example.jl"
exec_julia("jl_example.init()") # init module jl_example in jl_example.jl

from jl_example import jl_add, jl_mat_mul, jl_fft
