from jnumpy import init_jl, init_project

init_jl()
init_project(__file__)

from _extension import (
    jl_not,
    int_add,
    float_add,
    complex_mul_two,
    str_concat,
    tuple_return,
    mat_mul,
    set_zero,
    jl_fft,
)
