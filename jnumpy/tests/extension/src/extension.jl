module extension

using TyPython
using TyPython.CPython
using FFTW

@export_py function jl_not(a::Bool)::Bool
    return !a
end

@export_py function int_add(a::Int, b::Int)::Int
    return a + b
end

@export_py function float_add(a::AbstractFloat, b::AbstractFloat)::AbstractFloat
    return a + b
end

@export_py function complex_mul_two(x::Complex)::Complex
    return 2 * x
end

@export_py function str_concat(a::String, b::String)::String
    return a * b
end

@export_py function tuple_return(a::Int, b::String)::Tuple
    return a, b
end

@export_py function mat_mul(a::AbstractArray, b::AbstractArray)::Array
    return a * b
end

@export_py function set_zero(a::StridedVector)::Nothing
    a[1] = zero(eltype(a))
    return nothing
end

@export_py function jl_fft(x::StridedVector)::StridedVector{ComplexF64}
    return FFTW.fft(x)
end


function init()
    @export_pymodule _extension begin
        jl_not = Pyfunc(jl_not)
        int_add = Pyfunc(int_add)
        float_add = Pyfunc(float_add)
        complex_mul_two = Pyfunc(complex_mul_two)
        str_concat = Pyfunc(str_concat)
        tuple_return = Pyfunc(tuple_return)
        mat_mul = Pyfunc(mat_mul)
        set_zero = Pyfunc(set_zero)
        jl_fft = Pyfunc(jl_fft)
    end
end

precompile(init, ())

end