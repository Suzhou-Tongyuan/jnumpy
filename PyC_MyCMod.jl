module PyC_MyCMod
using TyPython
using TyPython.CPython
using MKL
using Primes
import FFTW
FFTW.set_num_threads(4)
TyPython.CPython.init()

@export_py function f(x::Int, y::Int)::Int
    x + y
end

@export_py function array_func(
    x::StridedVector{Float64},
    y::StridedVector{Int})::Float64
    s = zero(Float64)
    for i in 1:min(length(x), length(y))
        s += x[i] + y[i]
    end
    return s
end

const FFT1D_PlanType = Base._return_type(FFTW.plan_fft, (StridedVector{ComplexF64}, Int))
const plan1Ds = Dict{Tuple{DataType, NTuple{N, Int} where N}, FFT1D_PlanType}()

"""
1-D FFT
"""
@export_py function jfft(
        x::Union{StridedVector{Float64}, StridedVector{ComplexF64}})::StridedVector{ComplexF64}
    x = collect(ComplexF64, x)
    plan = get!(plan1Ds, (typeof(x), size(x))) do
        FFTW.plan_fft(copy(x), 1; flags=FFTW.MEASURE)
    end
    return plan * x
end

@export_py function jl_array_cache_pool()::Tuple{Int, Int}
    (length(CPython.G_arrayinfo), length(CPython.G_arrayinfo_unused_slots))
end

precompile(jfft, (Vector{Float64}, ))
precompile(jfft, (Vector{ComplexF64}, ))

@export_pymodule MyCMod begin
    array_func = Pyfunc(array_func)
    scalar_func = Pyfunc(f)
    fft = Pyfunc(jfft)
    jl_array_cache_pool = Pyfunc(jl_array_cache_pool)
    value = 1 # auto convert to py
end

end