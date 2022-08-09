from jnumpy import init_jl, init_project

init_jl()
init_project(__file__)

from _fast_fft import jl_fft
