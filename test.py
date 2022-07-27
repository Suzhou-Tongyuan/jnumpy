import os
os.environ['RAWPY_JL_OPTS'] = "--project" # + "--compile=min -O0"
from jnumpy import init_jl, exec_julia, include_src
import jnumpy as np
init_jl()
include_src('PyC_MyCMod.jl')

import MyCMod # type: ignore
import numpy as np
import scipy.fft as scipy_fft
xs = np.random.random(50000)  # .astype(np.complex128)
import gc

MyCMod.fft(xs)
print("scipy fft")
%timeit scipy_fft.fft(xs)
gc.collect()
print("jnumpy(tongyuan) fft")
%timeit MyCMod.fft(xs)

xs = np.random.random(50000).astype(np.complex128)
print("总元素:", len(xs), "结果相同:", sum(np.isclose(MyCMod.fft(xs), scipy_fft.fft(xs))))


