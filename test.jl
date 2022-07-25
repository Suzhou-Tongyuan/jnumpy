using RawPython
using RawPython.CPython
RawPython.CPython.init()
np = CPython.get_numpy()

println(CPython.G_PyBuiltin.__import__(py_cast(Py, "numpy")))
gc = CPython.G_PyBuiltin.__import__(py_cast(Py, "gc"))
np = get_numpy()
gc.collect()

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

@export_py function f(x::Int, y::Int)::Int
    x + y
end

@export_py function array_func(
    x::StridedVector{Float64},
    y::StridedVector{Float64})::Float64
    s = zero(Float64)
    for i in min(length(x), length(y))
        s += x[i] + y[i]
    end
    return s
end


@export_pymodule MyCExtensionInJulia begin
    array_func = Pyfunc(array_func)
    scalar_func = Pyfunc(f)
    value = 1 # auto convert to py
end






# gc.collect()
# pyfunc = Pyfunc(array_func)
# println(pyfunc(np.random.random(py_cast(Py, 10)), np.random.random(py_cast(Py, 9))))



# println(CPython.G_PyBuiltin.__import__(py_cast(Py, "sys")).modules[py_cast(Py, "MyCExtensionInJulia")])
# println(CPython.G_PyBuiltin.__import__(py_cast(Py, "MyCExtensionInJulia")))

# println(CPython.py_coerce(Tuple{Int, Int}, Py(CPython.pyunsafe_eval_expr_str("(100, '222')"))))
# println(pyfunc(py_cast(Py, 1), py_cast(Py, 1)))
# println(CPython.py_isnull(o))

# using BenchmarkTools
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
