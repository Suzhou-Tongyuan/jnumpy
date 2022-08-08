module jl_example

using TyPython
using TyPython.CPython
using FFTW
CPython.init()

@export_py function add(a::Int, b::Int)::Int
    return a + b
end

@export_py function mat_mul(a::StridedArray, b::StridedArray)::StridedArray
    return a * b
end

@export_py function myfft(x::StridedArray{Float64})::StridedArray{ComplexF64}
    return FFTW.fft(x)
end


function init()
    @export_pymodule jl_example begin
        jl_add = Pyfunc(add)
        jl_mat_mul = Pyfunc(mat_mul)
        jl_fft = Pyfunc(myfft)
    end
end

end