from jnumpy import init_jl, exec_julia, include_src

init_jl()
include_src("extension.jl", __file__)
exec_julia("extension.init()")

from extension import jl_not, int_add, float_add, complex_mul_two, str_concat, tuple_return, mat_mul, set_zero, jl_fft
