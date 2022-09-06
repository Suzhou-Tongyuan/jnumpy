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

@testset "jl -> JuliaRaw" begin
    @test pyisjl(pya)
    @test !(pyisjl(py_cast(Py, 1)))
    @test pya._jl_deserialize(pya._jl_serialize()).x == py_cast(Py, 1)
    @test repr(pya) == "Py(<jl Point(1, 2)>)"
    @test pya.__class__ == CPython.G_JNUMPY.JuliaRaw
    @test pya.x == py_cast(Py, 1)
    @test py_cast(Bool, pya.__mul__(pya).__eq__(py_cast(Py, Point(1; y=4))))
    @test pyPoint(py_cast(Py, 1); y=py_cast(Py, 2)).y == py_cast(Py, 2)
    pya.x = py_cast(Py, 3)
    @test a.x == 3
    @test py_cast(Int, pya.__dir__().__len__()) > 0
    @test_throws CPython.PyException pya.__add__(pya)
end