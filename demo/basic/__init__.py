from jnumpy import init_jl, exec_julia, include_src

init_jl()
include_src("core/basic.jl", __file__) # use a relative path
exec_julia("basic.init()") # init module basic in core/basic.jl

from basic import jl_int_add, jl_mat_mul
