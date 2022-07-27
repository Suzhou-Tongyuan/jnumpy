import os
os.environ['RAWPY_JL_OPTS'] = "--project" # + "--compile=min -O0"
from jnumpy import init_jl, exec_julia, include_src
import jnumpy as np
init_jl()
include_src('PyC_fast_kmeans.jl')
import _fast_kmeans # type: ignore

def kmeans(x: np.ndarray, n_clusters=3):
    assignments, centers = _fast_kmeans.kmeans(x.T, n_clusters)
    return assignments, centers.T
