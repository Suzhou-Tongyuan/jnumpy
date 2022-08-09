from jnumpy import init_jl, init_project
import jnumpy as np

init_jl()
init_project(__file__)

from _fast_kmeans import _fast_kmeans

def jl_kmeans(x: np.ndarray, n_clusters: int):
    assignments, center = _fast_kmeans(x.T, n_clusters)
    return assignments, center.T
