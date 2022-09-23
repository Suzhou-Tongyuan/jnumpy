mutable struct Point
    x
    y
    function Point(x; y)
        return new(x, y)
    end
end
a = Point(1; y=2)
pya = py_cast(Py, a)
Base.:*(a::Point, b::Point) = Point(a.x*b.x; y=(a.y*b.y))
Base.:(==)(a::Point, b::Point) = (a.x == b.x && a.y == b.y)
pyPoint = py_cast(Py, Point)
function test_dealloc()
    temp = py_cast(Py, Point(rand(); y=rand()))
    nothing
end
Base.firstindex(p::Point) = 1
Base.lastindex(p::Point) = 2
function Base.getindex(p::Point, i::Int)
    1 <= i <= 2 || throw(BoundsError(p, i))
    i == 1 ? p.x : p.y
end
function Base.setindex!(p::Point, v, i::Int)
    1 <= i <= 2 || throw(BoundsError(p, i))
    if i == 1
        p.x = v
    else
        p.y = v
    end
end

@testset "JuliaBase" begin
    @test pyisjl(pya)
    @test !(pyisjl(py_cast(Py, 1)))
    reduce_pya = pya.__reduce__()
    @test reduce_pya[py_cast(Py, 0)](reduce_pya[py_cast(Py, 1)][py_cast(Py, 0)]).x == py_cast(Py, 1)
    test_dealloc()
    GC.gc()
    @test length(CPython.PYJLFREEVALUES) > 0
end

@testset "JuliaRaw" begin
    @test repr(pya) == "Py(<jl Point(1, 2)>)"
    @test pya.x == py_cast(Py, 1)
    @test py_cast(Bool, pya.__mul__(pya).__eq__(py_cast(Py, Point(1; y=4))))
    @test pyPoint(py_cast(Py, 1); y=py_cast(Py, 2)).y == py_cast(Py, 2)
    pya.x = py_cast(Py, 3)
    @test a.x == 3
    @test py_cast(Int, pya.__dir__().__len__()) > 0
    @test py_cast(Int, py_cast(Py, CPython).__dir__().__len__()) > 0
    @test py_cast(String, py_cast(Py, CPython).__name__) == "CPython"
    @test_throws CPython.PyException pya.__add__(pya)
    @test pya[py_cast(Py, 1)] == py_cast(Py, 3)
    pya[py_cast(Py, 1)] = py_cast(Py, 1)
    @test pya[py_cast(Py, 1)] == py_cast(Py, 1)
end