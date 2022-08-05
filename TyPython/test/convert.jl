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
    @test py_coerce(Int, py_cast(Py, 1.1)) == 1
    @test_throws CPython.PyException py_coerce(Int, py_cast(Py, "abc"))
    @test py_coerce(Float64, py_cast(Py, 1.1)) == 1.1
    @test_throws CPython.PyException py_coerce(Float64, py_cast(Py, "abc"))
    @test py_coerce(ComplexF64, py_cast(Py, 1+0.5im)) == 1+0.5im
    @test_throws CPython.PyException py_coerce(ComplexF64, py_cast(Py, "abc"))
    @test py_coerce(ComplexF64, py_cast(Py, 1+0.5im)) == 1+0.5im
    @test_throws CPython.PyException py_coerce(ComplexF64, py_cast(Py, "abc"))
    @test py_coerce(String, py_cast(Py, "äbc")) == "äbc"
    @test py_coerce(Array, py_cast(Py, [1 2; 3 4])) == [1 2; 3 4]
    @test py_coerce(Array, py_cast(Py, [1 2; 3 4]).transpose()) == [1 3; 2 4]
    @test_throws CPython.PyException py_coerce(Array, py_cast(Py, "abc"))
end

@testset "py_cast" begin
    a = py_cast(Py, true)
    @test py_cast(Bool, a)
    b = py_cast(Py, "abc")
    @test py_cast(String, b) == "abc"
end


