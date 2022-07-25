import os
os.environ['RAWPY_JL_OPTS'] = "--compile=min -O0 --project"
from jnumpy import init_jl, eval_jl
init_jl()
eval_jl(r'''
try
    using RawPython.CPython
    CPython.WITH_GIL() do
        include("test.jl")
    end    
catch e
    Base.showerror(stderr, e, catch_backtrace())
end
''')

import MyCExtensionInJulia # type: ignore
import numpy as np

print(
    MyCExtensionInJulia.scalar_func(1, 2))
print(MyCExtensionInJulia.array_func(np.random.random(30), np.random.random(20)))

