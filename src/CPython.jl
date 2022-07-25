module CPython
import RawPython.C
import RawPython.Utils: capure_stdout, unroll_do!
export py_builtin_get, py_throw, WITH_GIL, GILNoRaise
export py_cast, py_coerce
export Py

const CF_RAWPY_MODE = "RAWPY_MODE"
const CF_RAWPY_PY_APIPTR = "RAWPY_PY_APIPTR"
const CF_RAWPY_PY_DLL = "RAWPY_PY_DLL"
const CF_RAWPY_MODE_PYTHON = "PYTHON-BASED"
const CF_RAWPY_MODE_JULIA = "JULIA-BASED"

function is_calling_julia_from_python()
    result = get!(ENV, CF_RAWPY_MODE) do 
        CF_RAWPY_MODE_JULIA
    end
    if result == CF_RAWPY_MODE_PYTHON
        return true
    end
    return false
end

include("CPython.Defs.jl")
include("CPython.APIs.jl")
include("CPython.Boot.jl")
include("CPython.MRO.jl")

const G_ATTR_SYM_MAP = Dict{Symbol, Py}()
const G_OB_POOL = Any[]

let ln = LineNumberNode((@__LINE__), Symbol(@__FILE__))
    for (fn, ft) in zip(fieldnames(PythonAPIStruct), fieldtypes(PythonAPIStruct))
        if ft <: CPyFunction
            expr_creat_fn = _support_ccall!(ln, ft)
            @eval $expr_creat_fn
        end
    end
end

include("CPython.NumPy.jl")
include("CPython.Julia.jl")

function __init__()
    empty!(G_OB_POOL)
    empty!(G_ATTR_SYM_MAP)
    __init_numpy__()
    __init_julia_wrap__()
end

end