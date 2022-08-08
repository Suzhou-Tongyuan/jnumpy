using TyPython
using TyPython.CPython
CPython.init()

import TyPython.CPython: py_cast, py_coerce, Py, PyAPI
using Test

@testset "TyPython" begin
    @testset "pyexport" begin
        include("pyexport.jl")
    end

    @testset "convert" begin
        include("convert.jl")
    end
end