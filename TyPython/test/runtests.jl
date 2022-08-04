using TyPython
using TyPython.CPython
CPython.init()

import TyPython.CPython: py_cast, Py, PyAPI
using Test

@testset "pyexport" begin
    include("pyexport.jl")
end

