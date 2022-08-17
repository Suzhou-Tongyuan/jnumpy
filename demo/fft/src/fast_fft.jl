module fast_fft

using TyPython
using TyPython.CPython
using MKL
import FFTW


const FFT1D_PlanType = Union{Base.return_types(FFTW.plan_fft, (StridedVector{ComplexF64}, Int))...}
const plan1Ds = Dict{Tuple{DataType, NTuple{N, Int} where N}, FFT1D_PlanType}()

"""
1-D FFT
"""
@export_py function jl_fft(x::StridedVector)::Vector{ComplexF64}
    x = collect(ComplexF64, x)
    plan = get!(plan1Ds, (typeof(x), size(x))) do
        FFTW.plan_fft(copy(x), 1; flags=FFTW.MEASURE)
    end
    return plan * x
end


function init()
    @export_pymodule _fast_fft begin
        jl_fft = Pyfunc(jl_fft)
    end
end

precompile(init, ())

end # end of module