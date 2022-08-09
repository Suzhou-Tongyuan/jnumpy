from jnumpy import init_jl, init_project

init_jl()
init_project(__file__)

from _basic import jl_int_add, jl_mat_mul
