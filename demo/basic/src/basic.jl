module basic

using TyPython
using TyPython.CPython

@export_py function int_add(a::Int, b::Int)::Int
    return a + b
end

@export_py function mat_mul(a::StridedArray, b::StridedArray)::StridedArray
    return a * b
end

function init()
    @export_pymodule _basic begin
        jl_int_add = Pyfunc(int_add)
        jl_mat_mul = Pyfunc(mat_mul)
    end
end

precompile(init, ())

end
