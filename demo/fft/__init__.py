from jnumpy import init_jl, exec_julia, include_src

init_jl()
include_src("core/fast_fft.jl", __file__) # use a relative path
exec_julia("fast_fft.init()") # init module fast_fft in core/fast_fft.jl

from fast_fft import jl_fft
