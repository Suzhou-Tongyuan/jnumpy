import jnumpy as np
from example import jl_add, jl_mat_mul, jl_fft

print("jl_add(1, 2) =", jl_add(1, 2), "\n")

a = np.array([[1, 2],[3, 4]])
b = np.array([[4, 3],[2, 1]])
print("jl_mat_mul(a,b) =\n", jl_mat_mul(a,b), "\n")

x = np.random.rand(2,2)
print("x =\n", x, "\n")
print("jl_fft(x) =\n", jl_fft(x), "\n")