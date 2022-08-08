from jnumpy import init_jl, exec_julia, include_src
import jnumpy as np

init_jl()
include_src("core/fast_kmeans.jl", __file__)
exec_julia("fast_kmeans.init()")

from fast_kmeans import _fast_kmeans

def jl_kmeans(x: np.ndarray, n_clusters: int):
    assignments, center = _fast_kmeans(x.T, n_clusters)
    return assignments, center.T
