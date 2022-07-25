using RawPython
using RawPython.CPython
RawPython.CPython.init()
np = get_numpy()

pyarr = np.ones(py_cast(Py, (3, 4)), order=py_cast(Py, "F"))
arr = CPython.from_ndarray(pyarr)

arr_new = copy(arr) .* 3
arr .= arr_new
println(pyarr)

xss = ones(10, 10)
x = py_coerce(Py, xss)

pyarr = np.arange(py_cast(Py, 18)).reshape(py_cast(Py, (2, 9))).T
println(pyarr)
println(pyarr.shape)
using BenchmarkTools
# @btime CPython.from_ndarray(pyarr)
# jlarr = CPython.from_ndarray(pyarr)
# println(jlarr)
# println(size(jlarr))
# jlarr[1, 1] = 100
# println(pyarr)


# using BenchmarkTools
# @btime py_coerce(Py, xss)
# GC.gc()
# println(length(CPython.G_arrayinfo_unused_slots))
# println(typeof(x))
# println(x)
