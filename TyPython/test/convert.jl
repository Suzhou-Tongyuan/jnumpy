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
    @test isnothing(py_coerce(Nothing, py_cast(Py, nothing)))

    @test py_coerce(Bool, py_cast(Py, true))
    @test !(py_coerce(Bool, py_cast(Py, false)))
    @test_throws CPython.PyException py_coerce(Bool, py_cast(Py, 1))

    @test py_coerce(Int, py_cast(Py, 1)) == 1
    x1 = py_coerce(Int32, py_cast(Py, 1))
    @test x1 isa Int32
    @test x1 == Int32(1)
    @test_throws CPython.PyException py_coerce(Int, py_cast(Py, 1.0))
    @test_throws CPython.PyException py_coerce(Int, py_cast(Py, "abc"))

    @test py_coerce(Float64, py_cast(Py, 1.1)) == 1.1
    x2 = py_coerce(Float32, py_cast(Py, 1.1))
    @test x2 isa Float32
    @test x2 == Float32(1.1)
    @test_throws CPython.PyException py_coerce(Float64, py_cast(Py, 1))
    @test_throws CPython.PyException py_coerce(Float64, py_cast(Py, "abc"))

    @test py_coerce(ComplexF64, py_cast(Py, 1+0.5im)) == 1+0.5im
    x3 = py_coerce(ComplexF32, py_cast(Py, 1+0.5im))
    @test x3 isa ComplexF32
    @test x3 == ComplexF32(1+0.5im)
    @test_throws CPython.PyException py_coerce(ComplexF64, py_cast(Py, 1.0))
    @test_throws CPython.PyException py_coerce(ComplexF64, py_cast(Py, "abc"))

    @test py_coerce(Tuple{Int, String}, py_cast(Py, (1, "a"))) == (1, "a")
    @test_throws CPython.PyException py_coerce(Tuple{Int, String}, py_cast(Py, (1, 1)))
    @test_throws CPython.PyException py_coerce(Tuple{Int, String}, py_cast(Py, 1))
end

@testset "py_cast" begin
    @test isnothing(py_cast(Nothing, py_cast(Py, nothing)))
    @test_throws CPython.PyException py_cast(Nothing, py_cast(Py, "abc"))

    @test py_cast(Bool, py_cast(Py, true))
    @test !(py_cast(Bool, py_cast(Py, false)))
    @test py_cast(Bool, py_cast(Py, 1))

    @test py_cast(Int, py_cast(Py, 1)) == 1
    @test py_cast(Int, py_cast(Py, 1.1)) == 1
    @test_throws CPython.PyException py_cast(Int, py_cast(Py, "abc"))

    @test py_cast(Float64, py_cast(Py, 1.0)) === 1.0
    @test py_cast(Float64, py_cast(Py, 1)) === 1.0
    @test_throws CPython.PyException py_cast(Float64, py_cast(Py, "abc"))

    @test py_cast(ComplexF64, py_cast(Py, 1+0.5im)) === 1+0.5im
    @test py_cast(ComplexF64, py_cast(Py, 1)) === 1.0+0.0im
    @test_throws CPython.PyException py_cast(Float64, py_cast(Py, "abc"))

    @test py_cast(ComplexF64, py_cast(Py, 1+0.5im)) === 1+0.5im
    @test py_cast(ComplexF64, py_cast(Py, 1)) === 1.0+0.0im
    @test_throws CPython.PyException py_cast(Float64, py_cast(Py, "abc"))

    @test py_cast(Tuple{Int, String}, py_cast(Py, (1, "a"))) == (1, "a")
    @test_throws CPython.PyException py_cast(Tuple{Int, String}, py_cast(Py, (1, 1)))
    @test_throws ErrorException py_cast(Tuple{Int}, py_cast(Py, (1, 1)))
    @test_throws CPython.PyException py_cast(Tuple{Int, String}, py_cast(Py, 1))

    # in these cases py_cast falls back to py_coerce
    @test py_cast(String, py_cast(Py, "äbc")) == "äbc"
    np = CPython.get_numpy()
    a = [1 2 3; 3 4 5]
    @testset "Array -> ndarray" begin
        py_a = py_cast(Py, a)
        @test py_cast(Array, py_a) == a
        @test py_cast(Bool, py_a.flags.f_contiguous)
        @test !(py_cast(Bool, py_a.flags.c_contiguous))
        x = py_cast(Matrix{Int32}, py_a)
        @test x isa Matrix{Int32}
        @test x == Int32.(a)
    end
    @testset "SubArray -> ndarray" begin
        a1 = @views a[1:1, 1:3] # not f contiguous
        py_a1 = py_cast(Py, a1)
        @test py_cast(Array, py_a1) == a1
        @test !(py_cast(Bool, py_a1.flags.f_contiguous))
        @test !(py_cast(Bool, py_a1.flags.c_contiguous))
        a2 = @views a[1:2, 1:2] # f contiguous
        py_a2 = py_cast(Py, a2)
        @test py_cast(Array, py_a2) == a2
        @test py_cast(Bool, py_a2.flags.f_contiguous)
        @test !(py_cast(Bool, py_a2.flags.c_contiguous))
    end
    @testset "Transpose -> ndarray" begin
        py_aT = py_cast(Py, transpose(a))
        @test py_cast(Array, py_aT) == collect(transpose(a))
        @test py_cast(Bool, py_aT.flags.c_contiguous)
        @test !(py_cast(Bool, py_aT.flags.f_contiguous))
        @test py_cast(Array, py_cast(Py, a')) == collect(a')
        @test py_cast(Bool, py_cast(Py, a').flags.f_contiguous) # copy
    end
    @testset "PermutedDimsArray -> ndarray" begin
        b = PermutedDimsArray(rand(Int, (2, 3, 4)), (3, 2, 1))
        py_b = py_cast(Py, b)
        @test py_cast(Bool, py_b.flags.c_contiguous)
        @test !(py_cast(Bool, py_b.flags.f_contiguous))
        @test py_cast(NTuple{3, Int}, py_b.shape) == (4, 3, 2)
        @test py_cast(PermutedDimsArray, py_b) == b
        b[1] = 0
        @test py_cast(Int64, py_b.__getitem__(py_cast(Py, (0,0,0)))) == 0
        py_c = np.random.random(py_cast(Py, (2, 3, 4))).transpose(py_cast(Py, (2, 0, 1)))
        @test py_cast(Array, py_c) isa Array{Float64, 3}
    end
    @testset "Array with shpae (1,k,1,1)" begin
        d = rand(1, 10, 1, 1)
        py_d = py_cast(Py, d)
        @test py_cast(Bool, py_d.flags.c_contiguous)
        @test py_cast(Bool, py_d.flags.f_contiguous)
        d[1] = 0.0
        @test py_cast(Float64, py_d.__getitem__(py_cast(Py, (0,0,0,0)))) == 0.0
        py_dT = py_cast(Py, PermutedDimsArray(d, (4, 3, 1, 2)))
        @test py_cast(Bool, py_dT.flags.c_contiguous)
        @test py_cast(Bool, py_dT.flags.f_contiguous)
        d[2] = 1.0
        @test py_cast(Float64, py_dT.__getitem__(py_cast(Py, (0,0,0,1)))) == 1.0
        e = np.random.random(py_cast(Py, (1, 10, 1, 1)))
        @test py_cast(AbstractArray, e) isa Array
    end
    @testset "SubArray and ReinterpretArray" begin
        x = rand(4, 4)
        x_sub = @view x[2:4, :]
        py_x_sub = py_cast(Py, x_sub)
        @test py_cast(Tuple{Int, Int}, py_x_sub.shape) == (3, 4)
        @test py_cast(Float64, py_x_sub[py_cast(Py, (0,0))]) == x_sub[1, 1]
        @test !py_cast(Bool, py_x_sub.flags.c_contiguous)
        @test !py_cast(Bool, py_x_sub.flags.f_contiguous)
        x_re = reinterpret(ComplexF64, x)
        py_x_re = py_cast(Py, x_re)
        @test py_cast(Tuple{Int, Int}, py_x_re.shape) == (2, 4)
        @test pointer(py_cast(Array, py_x_re)) == pointer(x)
    end
    @test_throws CPython.PyException py_cast(Array, py_cast(Py, "abc"))
end

@testset "kwargs" begin
    py_int = CPython.get_py_builtin().int
    x = py_int(py_cast(Py, "010"), base=py_cast(Py, 16))
    @test py_coerce(Int, x) == 16
end