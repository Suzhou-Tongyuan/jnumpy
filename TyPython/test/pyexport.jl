module test_export

using TyPython
using TyPython.CPython
CPython.init()

@export_py function jl_not(a::Bool)::Bool
    return !a
end

@export_py function int_add(a::Int, b::Int)::Int
    return a + b
end

@export_py function float_add(a::AbstractFloat, b::AbstractFloat)::AbstractFloat
    return a + b
end

@export_py function complex_mul_two(x::Complex)::Complex
    return 2 * x
end

@export_py function str_concat(a::String, b::String)::String
    return a * b
end

@export_py function tuple_return(a::Int, b::String)::Tuple
    return a, b
end

@export_py function mat_mul(a::StridedArray, b::StridedArray)::StridedArray
    return a * b
end

@export_py function set_zero(a::StridedArray)::Nothing
    a[1] = zero(eltype(a))
    return nothing
end

function init()
    @export_pymodule test_export begin
        jl_not = Pyfunc(jl_not)
        int_add = Pyfunc(int_add)
        float_add = Pyfunc(float_add)
        complex_mul_two = Pyfunc(complex_mul_two)
        str_concat = Pyfunc(str_concat)
        tuple_return = Pyfunc(tuple_return)
        mat_mul = Pyfunc(mat_mul)
        set_zero = Pyfunc(set_zero)
    end
end

end # end module


test_export.init()
dict = TyPython.CPython.G_PyBuiltin.dict
const src_header = raw"""
from test_export import jl_not, int_add, float_add, complex_mul_two, str_concat, tuple_return, mat_mul, set_zero
import numpy as np
"""

function test_py_code(src_code::String)
    src_code = src_header * src_code
    TyPython.CPython.G_PyBuiltin.exec(py_cast(Py, src_code), dict())
end

@testset "not" begin
    src_code =  raw"""
assert jl_not(False)
"""
    @test test_py_code(src_code) == PyAPI.Py_None
end

@testset "test_add" begin
    src_code =  raw"""
assert int_add(1, 2)==3
np.testing.assert_almost_equal(float_add(1.0, 2.0), 3.0)
"""
    @test test_py_code(src_code) == PyAPI.Py_None
end

@testset "test_complex()" begin
    src_code =  raw"""
x = complex(1.0, 2.1)
np.testing.assert_almost_equal(complex_mul_two(x), 2*x)
"""
    @test test_py_code(src_code) == PyAPI.Py_None
end

@testset "test_str_concat()" begin
    src_code =  raw"""
assert str_concat("a", "b") == "ab"
"""
    @test test_py_code(src_code) == PyAPI.Py_None
end

@testset "test_tuple_return()" begin
    src_code =  raw"""
assert tuple_return(1, "a") == (1, "a")
"""
    @test test_py_code(src_code) == PyAPI.Py_None
end

@testset "test_set_zero()" begin
    src_code =  raw"""
x = np.random.rand(2)
set_zero(x)
np.testing.assert_almost_equal(x[0], 0.0)
"""
    @test test_py_code(src_code) == PyAPI.Py_None
end

@testset "test_mat_mul()" begin
    src_code =  raw"""
dtype_list = [np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
    np.complex64, np.complex128]
for dtype in dtype_list:
    x = np.asarray([[1.2, 3.4],[2.3, 5.6]], dtype=dtype)
    y = np.asarray([[7.8, 5e-3],[6.75, 8.234]], dtype=dtype)
    actual = mat_mul(x, y)
    desired = x @ y
    np.testing.assert_array_almost_equal(actual, desired, decimal=5)
"""
    @test test_py_code(src_code) == PyAPI.Py_None
end