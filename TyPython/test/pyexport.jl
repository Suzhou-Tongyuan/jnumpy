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

funcs = CPython.py_import("test_export")
np = CPython.py_import("numpy")
assert_almost_equal = np.testing.assert_almost_equal
assert_array_almost_equal = np.testing.assert_array_almost_equal

function test_py_code(src_code::String)
    src_code = src_header * src_code
    TyPython.CPython.G_PyBuiltin.exec(py_cast(Py, src_code), dict())
end

@testset "not" begin
    @test funcs.jl_not(py_cast(Py, false)) == py_cast(Py, true)
end

@testset "add" begin
    @test funcs.int_add(py_cast(Py, 1), py_cast(Py, 2)) == py_cast(Py, 3)
    py_float_three = funcs.float_add(py_cast(Py, 1.0), py_cast(Py, 2.0))
    @test assert_almost_equal(py_float_three, py_cast(Py, 3.0)) == PyAPI.Py_None
end

@testset "complex" begin
    x = py_cast(Py, 1.0+2.1im)
    two = py_cast(Py, 2)
    @test assert_almost_equal(funcs.complex_mul_two(x), x.__mul__(two)) == PyAPI.Py_None
end

@testset "str_concat" begin
    a = py_cast(Py, "a")
    b = py_cast(Py, "b")
    ab = py_cast(Py, "ab")
    @test funcs.str_concat(a, b) == ab
end

@testset "tuple_return" begin
    one = py_cast(Py, 1)
    a = py_cast(Py, "a")
    @test funcs.tuple_return(one, a) == py_cast(Py, (one, a))
end

@testset "set_zero" begin
    x = np.random.rand(py_cast(Py, 2))
    funcs.set_zero(x)
    @test assert_almost_equal(x[py_cast(Py, 0)], py_cast(Py, 0.0)) == PyAPI.Py_None
end

@testset "mat_mul" begin
    dtype_list = Py[np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128]
    x0 = py_cast(Py, [1.2 3.4; 2.3 5.6])
    y0 = py_cast(Py, [7.8 5e-3; 6.75 8.234])

    for dtype in dtype_list
        x = np.asarray(x0, dtype=dtype)
        y = np.asarray(y0, dtype=dtype)
        actual = funcs.mat_mul(x, y)
        desired = np.matmul(x, y)
        @test assert_array_almost_equal(actual, desired, decimal=py_cast(Py, 5)) == PyAPI.Py_None
    end
end

@testset "py_exec" begin
    dict = TyPython.CPython.G_PyBuiltin.dict
    res = TyPython.CPython.G_PyBuiltin.exec(py_cast(Py, "1+1"), dict())
    @test res == PyAPI.Py_None
end