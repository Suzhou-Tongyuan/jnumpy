import Libdl

struct UnsafeNew end
struct NewReference end
struct BorrowReference end

mutable struct Py
    ptr :: C.Ptr{PyObject}

    function Py(::BorrowReference, ptr::C.Ptr{PyObject})
        return new(ptr)
    end
    function Py(::UnsafeNew, ptr::C.Ptr{PyObject}=Py_NULLPTR)
        self = new(ptr)
        finalizer(self) do x
            if RT_is_initialized()
                WITH_GIL(GILNoRaise()) do
                    PyAPI.Py_DecRef(x)
                end
            end
        end
        return self
    end
end

Py(ptr::C.Ptr{PyObject}) = Py(UnsafeNew(), ptr)
const G_PyBuiltin = Py(UnsafeNew())

function get_py_builtin()
    return G_PyBuiltin
end

function Py(::NewReference, ptr::C.Ptr{PyObject})
    PyAPI.Py_IncRef(ptr)
    return Py(UnsafeNew(), ptr)
end

unsafe_unwrap(x::Py) = getfield(x, :ptr)
unsafe_unwrap(x::C.Ptr{PyObject}) = x

# TODO: steal
function unsafe_set!(x::Py, p::C.Ptr{PyObject})
    if unsafe_unwrap(x) !== Py_NULLPTR
        error("Py object already set")
    end
    setfield!(x, :ptr, p)
    nothing
end

mutable struct PythonAPIStruct
    Py_Initialize::cfunc_t(Cvoid)
    Py_InitializeEx::cfunc_t(Cint, Cvoid)
    Py_IsInitialized::cfunc_t(Cint)
    Py_Finalize::cfunc_t(Cvoid)
    Py_FinalizeEx::cfunc_t(Cint)
    Py_DecRef::cfunc_t(C.Ptr{PyObject}, Cvoid) # no except
    Py_IncRef::cfunc_t(C.Ptr{PyObject}, Cvoid) # no except
    PyGILState_Ensure::cfunc_t(PyGILState) # no except
    PyGILState_Release::cfunc_t(PyGILState, Cvoid) # no except

    PyObject_Not::cfunc_t(C.Ptr{PyObject}, Except(-1, Cint)) # except -1
    PyObject_IsTrue::cfunc_t(C.Ptr{PyObject}, Except(-1, Cint)) # except -1
    PyObject_Call::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject}))
    PyObject_CallObject::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject}))
    PyObject_CallFunctionObjArgs_variant_1arg::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject}))
    PyObject_CallFunctionObjArgs_variant_0arg::cfunc_t(C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject}))
    PyObject_Hash::cfunc_t(C.Ptr{PyObject}, Except(-1, Py_hash_t)) # except -1
    PyObject_Bytes::cfunc_t(C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_Str::cfunc_t(C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_Repr::cfunc_t(C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_Length::cfunc_t(C.Ptr{PyObject}, Except(-1, Py_ssize_t)) # except -1
    PyObject_RichCompare::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Cint, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_SetAttr::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, C.Ptr{PyObject}, Except(-1, Cint)) # except -1
    PyObject_GetAttr::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_HasAttr::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Cint) # never fail
    PyObject_SetItem::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, C.Ptr{PyObject}, Except(-1, Cint)) # except -1
    PyObject_GetItem::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_GetIter::cfunc_t(C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_Dir::cfunc_t(C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyObject_IsInstance::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Except(-1, Cint)) # except -1
    PyUnicode_FromString::cfunc_t(Cstring, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyUnicode_AsUTF8AndSize::cfunc_t(C.Ptr{PyObject}, Ptr{Py_ssize_t}, Except(C_NULL, Ptr{Cchar}))
    PyUnicode_FromStringAndSize::cfunc_t(Cstring, Py_ssize_t, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyErr_Print::cfunc_t(Cvoid) # no except
    PyErr_Occurred::cfunc_t(C.Ptr{PyObject}) # not set -> NULL
    PyErr_SetString::cfunc_t(C.Ptr{PyObject}, Cstring, Cvoid) # no except
    PyErr_SetObject::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Cvoid) # no except

    PyErr_Fetch::cfunc_t(C.Ptr{C.Ptr{PyObject}}, C.Ptr{C.Ptr{PyObject}}, C.Ptr{C.Ptr{PyObject}}, Cvoid) # no except
    PyErr_NormalizeException::cfunc_t(C.Ptr{C.Ptr{PyObject}}, C.Ptr{C.Ptr{PyObject}}, C.Ptr{C.Ptr{PyObject}}, Cvoid) # no except
    PyException_SetTraceback::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, Cvoid) # no except
    PyErr_Clear::cfunc_t(Cvoid) # no except
    PyExc_TypeError::C.Ptr{C.Ptr{PyObject}}
    PyExc_IndexError::C.Ptr{C.Ptr{PyObject}}
    PyExc_KeyError::C.Ptr{C.Ptr{PyObject}}
    PyExc_ValueError::C.Ptr{C.Ptr{PyObject}}

    PyTuple_SetItem::cfunc_t(C.Ptr{PyObject}, Py_ssize_t, C.Ptr{PyObject}, Except(-1, Cint)) # except -1
    PyTuple_GetItem::cfunc_t(C.Ptr{PyObject}, Py_ssize_t, Except(Py_NULLPTR, C.Ptr{PyObject}))
    PyTuple_New::cfunc_t(Py_ssize_t, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyTuple_Size::cfunc_t(C.Ptr{PyObject}, Py_ssize_t)

    PyImport_ImportModule::cfunc_t(Cstring, Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL

    PyDict_New::cfunc_t(Except(Py_NULLPTR, C.Ptr{PyObject})) # except NULL
    PyDict_SetItem::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, C.Ptr{PyObject}, Except(-1, Cint)) # except -1
    PyDict_Type::C.Ptr{PyObject}

    PyLong_Type::C.Ptr{PyObject}
    PyFloat_Type::C.Ptr{PyObject}
    PyComplex_Type::C.Ptr{PyObject}
    PyTuple_Type::C.Ptr{PyObject}

    PyLong_AsLongLong::cfunc_t(C.Ptr{PyObject}, Clonglong) # except -1 ana error occurred
    PyLong_FromLongLong::cfunc_t(Clonglong, Except(Py_NULLPTR, C.Ptr{PyObject})) # except -1 ana error occurred
    PyFloat_AsDouble::cfunc_t(C.Ptr{PyObject}, Cdouble) # except -1.0 ana error occurred
    PyFloat_FromDouble::cfunc_t(Cdouble, Except(Py_NULLPTR, C.Ptr{PyObject}))
    PyComplex_AsCComplex::cfunc_t(C.Ptr{PyObject}, Py_complex) # except .real is -1.0 ana error occurred
    PyComplex_FromCComplex::cfunc_t(Py_complex, C.Ptr{PyObject})
    PyLong_AsSsize_t::cfunc_t(C.Ptr{PyObject}, Py_ssize_t) # except -1 ana error occurred
    PyNumber_Check::cfunc_t(C.Ptr{PyObject}, Cint)
    PyNumber_Long::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject})

    PyEval_EvalCode::cfunc_t(C.Ptr{PyObject}, C.Ptr{PyObject}, C.Ptr{PyObject}, Except(Py_NULLPTR, C.Ptr{PyObject}))
    Py_CompileString::cfunc_t(Cstring, Cstring, Cint, Except(Py_NULLPTR, C.Ptr{PyObject}))

    PyCapsule_Type::C.Ptr{PyObject}
    PyCapsule_SetDestructor::cfunc_t(C.Ptr{PyObject}, Ptr{Cvoid}, Cint)
    PyCapsule_GetPointer::cfunc_t(C.Ptr{PyObject}, Cstring, Ptr{Cvoid})
    PyCapsule_GetName::cfunc_t(C.Ptr{PyObject}, Cstring)
    PyCapsule_SetName::cfunc_t(C.Ptr{PyObject}, Cstring, Cint)
    PyCapsule_New::cfunc_t(Ptr{Cvoid}, Cstring, Ptr{Cvoid}, C.Ptr{PyObject})
    PyObject_ClearWeakRefs::cfunc_t(C.Ptr{PyObject}, Cvoid)
    PyCFunction_NewEx::cfunc_t(Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Except(Py_NULLPTR, C.Ptr{PyObject}))
    Py_AtExit::cfunc_t(Ptr{Cvoid}, Cint)
    Py_None::Py
    Py_True::Py
    Py_False::Py
    PythonAPIStruct() = new()
end

const PyAPI = PythonAPIStruct()



"""
see `unsafe_broaden_type(::Type{C.Ptr{O}})` at CPython.Defs.jl,
where a compatible `cconvert` rule is required for `Py`.
"""
Base.cconvert(::Type{C.Ptr{PyObject}}, py::Py) = py
Base.unsafe_convert(::Type{C.Ptr{PyObject}}, py::Py) = unsafe_unwrap(py)
Base.cconvert(::Type{C.Ptr{C.Ptr{PyObject}}}, py::Ref{C.Ptr{PyObject}}) = py
Base.unsafe_convert(::Type{C.Ptr{C.Ptr{PyObject}}}, py::Ref{C.Ptr{PyObject}}) =
    reinterpret(C.Ptr{C.Ptr{PyObject}}, Base.unsafe_convert(Ptr{C.Ptr{PyObject}}, py))

struct GILNoRaise end

"""
Usage:
```
WITH_GIL() do
    ...
end
```

When performance is critical and you are sure that no exception will raise:

```
WITH_GIL(GILNoRaise()) do
    ...
end
```
"""
@inline function WITH_GIL(f, ::GILNoRaise)
    if is_calling_julia_from_python() && RT_is_initialized()
        g = PyAPI.PyGILState_Ensure()
        r = f()
        PyAPI.PyGILState_Release(g)
        return r
    end
    return f()
end

@inline function GIL_BEGIN()
    if is_calling_julia_from_python() && RT_is_initialized()
        return g = PyAPI.PyGILState_Ensure()
    end
end

@inline function GIL_END(g)
    if is_calling_julia_from_python() && RT_is_initialized()
        PyAPI.PyGILState_Release(g)
    end
end

@inline function WITH_GIL(f)
    if is_calling_julia_from_python() && RT_is_initialized()
        g = PyAPI.PyGILState_Ensure()
        try
            return f()
        finally
            PyAPI.PyGILState_Release(g)
        end
    end
    return f()
end

function endswith_decref(f, obs::C.Ptr{PyObject}...)
    try
        f(obs...)
    finally
        for ob in obs
            PyAPI.Py_DecRef(ob)
        end
    end
end
const Py_eval_input = 258  # since 2011
const Py_single_input = 256  # since 2011

pyunsafe_compile_expr_str(x::AbstractString) = convert(String, x)
function pyunsafe_compile_expr_str(x::String; filename::AbstractString="<string>")
    PyAPI.Py_CompileString(x, filename, Py_eval_input)
end

function pyunsafe_eval_expr_str(x::String; filename::AbstractString="<string>")
    local dict = C.Ptr{PyObject}(PyAPI.PyDict_Type)
    endswith_decref(
            pyunsafe_compile_expr_str(x; filename=filename),
            PyAPI.PyObject_CallObject(dict, Py_NULLPTR),
            PyAPI.PyObject_CallObject(dict, Py_NULLPTR)
        ) do code, globals, locals

        PyAPI.PyEval_EvalCode(code, globals, locals)
    end
end

function pyunsafe_fetch_error()
    t = Ref(Py_NULLPTR)
    v = Ref(Py_NULLPTR)
    b = Ref(Py_NULLPTR)
    PyAPI.PyErr_Fetch(t, v, b)
    if b[] === Py_NULLPTR && t[] === Py_NULLPTR
        error("PyErr_Fetch returned NULL")
    end
    PyAPI.PyErr_NormalizeException(t, v, b)
    if !py_isnull(b[])
        PyAPI.PyException_SetTraceback(v[], b[])
    end
    Py(t[]), Py(v[]), Py(b[])
end

"""
    PyException(x)

Wraps the Python exception `x` as a Julia `Exception`.
"""
mutable struct PyException <: Exception
    type::Py
    value::Py
    traceback::Py
end

function py_isnull(x::Py)
    unsafe_unwrap(x) === Py_NULLPTR
end

function py_isnull(x::C.Ptr{PyObject})
    x === Py_NULLPTR
end

@noinline function py_throw()
    if !RT_is_initialized()
        msg = capture_out() do
            PyAPI.PyErr_Print()
            PyAPI.PyErr_Clear()
        end
        error(msg)
    end

    local (t, v, b) = pyunsafe_fetch_error()
    if py_isnull(t)
        if py_isnull(v)
            error("no exception")
        else
            t_ptr = reinterpret(C.Ptr{PyObject}, unsafe_unwrap(v).type)
            PyAPI.Py_IncRef(t_ptr)
            t = Py(t_ptr)
        end
    end
    if py_isnull(v)
        v = G_PyBuiltin.None
    end
    if py_isnull(b)
        b = G_PyBuiltin.None
    end
    throw(PyException(t, v, b))
end

function Base.show(io::IO, e::Py)
    if py_isnull(e)
        Base.print(io, "Py(NULL)")
        return
    end
    x = Py(PyAPI.PyObject_Repr(e))
    GC.@preserve x begin
        size_ref = Ref(0)
        buf = PyAPI.PyUnicode_AsUTF8AndSize(x , size_ref)
        print(io, "Py(", Base.unsafe_string(buf, size_ref[]), ")")
    end
end

Base.showerror(io::IO, e::PyException) = _showerror(io, e, nothing, backtrace=false)
Base.showerror(io::IO, e::PyException, bt; backtrace=true) = _showerror(io, e, bt; backtrace=backtrace)
function _showerror(io::IO, e::PyException, bt; backtrace=true)
    traceback = Py(PyAPI.PyUnicode_FromString("traceback"))
    mod_traceback = G_PyBuiltin.__import__(traceback)
    mod_io = G_PyBuiltin.__import__(py_cast(Py, "io"))
    str_io = mod_io.StringIO()
    mod_traceback.print_exception(e.type, e.value, e.traceback, file=str_io)
    print(io, py_cast(String, str_io.getvalue()))
end


function py_seterror!(e::Py, msg::AbstractString)
    PyAPI.PyErr_SetObject(e, py_cast(Py, msg))
end

function get_refcnt(o:: Union{Py, C.Ptr{PyObject}})::Py_ssize_t
    return unsafe_unwrap(o).refcnt[]
end

function py_import(name::AbstractString)
    Py(PyAPI.PyImport_ImportModule(name))
end

