import jnumpy as np
from basic import jl_int_add, jl_mat_mul
from fft import jl_fft
from kmeans import jl_kmeans

print("jl_int_add(1, 2) =", jl_int_add(1, 2), "\n")

a = np.array([[1, 2],[3, 4]])
b = np.array([[4, 3],[2, 1]])
print("jl_mat_mul(a,b) =\n", jl_mat_mul(a,b), "\n")

x = np.array([0, 1, 2, 1])
print("x =\n", x, "\n")
print("jl_fft(x) =\n", jl_fft(x), "\n")

data = np.random.rand(10000, 500)
print("computing jl_kmeans(data, 3)\n")
assignments, center = jl_kmeans(data, 3)
print("assignments =\n", assignments, "\n")
print("center.shape =", center.shape)