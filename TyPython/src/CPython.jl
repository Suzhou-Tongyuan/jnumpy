module CPython
using MLStyle: @match, @switch
import LinearAlgebra
import TyPython.C
import TyPython.Utils: capture_out, unroll_do!
export get_py_builtin, py_throw, WITH_GIL, GILNoRaise
export py_cast, py_coerce
export Py

const G_IsInitialized = Ref(false)
const CF_TYPY_MODE = "TYPY_MODE"
const CF_TYPY_PY_APIPTR = "TYPY_PY_APIPTR"
const CF_TYPY_PY_DLL = "TYPY_PY_DLL"
const CF_TYPY_MODE_PYTHON = "PYTHON-BASED"
const CF_TYPY_MODE_JULIA = "JULIA-BASED"

function is_calling_julia_from_python()
    result = get!(ENV, CF_TYPY_MODE) do
        CF_TYPY_MODE_JULIA
    end
    if result == CF_TYPY_MODE_PYTHON
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
include("CPython.Dev.jl")


function __init__()
    G_IsInitialized[] = false
    empty!(G_OB_POOL)
    empty!(G_ATTR_SYM_MAP)
    __init_numpy__()
    __init_julia_wrap__()
end

end