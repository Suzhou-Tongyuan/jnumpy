@testset "hasproperty" begin
    a = py_cast(Py, 1)
    @test hasproperty(a, :bit_length)
    @test hasproperty(a, :real)
    @test hasproperty(a, :imag)
end

@testset "length" begin
    a = py_cast(Py, (1, 2, 3))
    @test length(a) == 3
end

@testset "getindex" begin
    a = py_cast(Py, (1, 2, 3))
    @test a[py_cast(Py, 0)] == py_cast(Py, 1)
    @test a[py_cast(Py, 1)] == py_cast(Py, 2)
    @test a[py_cast(Py, 2)] == py_cast(Py, 3)
end

@testset "py_array_get_index" begin
    A = py_cast(Py, [1 2; 3 4])
    @test A[py_cast(Py, 0), py_cast(Py, 0)] == py_cast(Py, 1)
    @test A[py_cast(Py, 1), py_cast(Py, 1)] == py_cast(Py, 4)
end

@testset "py_array_set_index" begin
    A = py_cast(Py, [1 2; 3 4])
    A[py_cast(Py, 0), py_cast(Py, 0)] = py_cast(Py, 0)
    @test A[py_cast(Py, 0), py_cast(Py, 0)] == py_cast(Py, 0)
end

@testset "compare" begin
    one = py_cast(Py, 1)
    two = py_cast(Py, 2)
    @test one == py_cast(Py, 1.0)
    @test one != two
    @test one < two
    @test one <= two
    @test two > one
    @test two >= one
end

@testset "py_dir" begin
    res = CPython.py_dir(py_cast(Py, 1))
    @test res[py_cast(Py, 0)] == py_cast(Py, "__abs__")
end

@testset "py_coerce" begin
    @test py_coerce(Int, py_cast(Py, 1)) == 1
    x1 = py_coerce(Int32, py_cast(Py, 1))
    @test x1 isa Int32
    @test x1 == Int32(1)
    @test_throws CPython.PyException py_coerce(Int, py_cast(Py, "abc"))

    @test py_coerce(Float64, py_cast(Py, 1.1)) == 1.1
    x2 = py_coerce(Float32, py_cast(Py, 1.1))
    @test x2 isa Float32
    @test x2 == Float32(1.1)
    @test_throws CPython.PyException py_coerce(Float64, py_cast(Py, "abc"))

    @test py_coerce(ComplexF64, py_cast(Py, 1+0.5im)) == 1+0.5im
    x3 = py_coerce(ComplexF32, py_cast(Py, 1+0.5im))
    @test x3 isa ComplexF32
    @test x3 == ComplexF32(1+0.5im)
    @test_throws CPython.PyException py_coerce(ComplexF64, py_cast(Py, "abc"))

    @test py_coerce(Tuple{Int, String}, py_cast(Py, (1, "a"))) == (1, "a")
    @test_throws CPython.PyException py_coerce(Tuple{Int, String}, py_cast(Py, (1, 1)))

    @test py_coerce(String, py_cast(Py, "äbc")) == "äbc"
    a = [1 2 3; 3 4 5]
    @test py_coerce(Array, py_cast(Py, a)) == [1 2 3; 3 4 5]
    @test py_coerce(Array, py_cast(Py, transpose(a))) == [1 3; 2 4; 3 5]
    x4 = py_coerce(Matrix{Int32}, py_cast(Py, Float32.(a)))
    @test x4 isa Matrix{Int32}
    @test x4 == Int32[1 2 3; 3 4 5]
    @test_throws CPython.PyException py_coerce(Array, py_cast(Py, "abc"))
    @test_throws MethodError py_cast(Py, a') # adjoint is unspported
end

@testset "py_cast" begin
    a1 = py_cast(Py, true)
    @test py_cast(Bool, a1)
    a2 = py_cast(Py, false)
    @test !py_cast(Bool, a2)
    a3 = py_cast(Py, true)
    @test py_cast(Bool, a3)

    b = py_cast(Py, "abc")
    @test py_cast(String, b) == "abc"
end


