struct Point
    x
    y
end
a = Point(1, 2)
pya = py_cast(Py, a)
Base.:*(a::Point, b::Point) = Point(a.x*b.x, a.y*b.y)
pyPoint = py_cast(Py, Point)

@testset "jl -> RawValue" begin
    @test repr(pya) == "Py(<jl Point(1, 2)>)"
    @test pya.__class__ == CPython.G_JNUMPY.RawValue
    @test pya.x == py_cast(Py, 1)
    @test pya.__mul__(pya) == py_cast(Py, Point(1, 4))
    @test pyPoint(py_cast(Py, 1), py_cast(Py, 2)).x == py_cast(Py, 1)
end