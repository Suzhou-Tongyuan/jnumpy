import os
os.environ['RAWPY_JL_OPTS'] = "--compile=min -O0 --project"
from jnumpy import init_jl, eval_jl;init_jl()

# x = eval_jl(r"""
# begin
#     using RawPython.CPython
#     CPython.from_ndarray
# end
# """)


