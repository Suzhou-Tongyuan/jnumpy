const Py_intptr_t = Cssize_t  # TODO: verify for portability

const NPY_ARRAY_C_CONTIGUOUS = Cint(0x0001)
const NPY_ARRAY_F_CONTIGUOUS = Cint(0x0002)
const NPY_ARRAY_ALIGNED = Cint(0x0100)
const NPY_ARRAY_NOTSWAPPED = Cint(0x0200)
const NPY_ARRAY_WRITEABLE = Cint(0x0400)
const NPY_ARR_HAS_DESCR = Cint(0x0800)

const TYPY_SUPPORTED_NO_COPY_NP_FLAG = NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE

function checkbit(x::Cint, bits::Cint)
    return (x & bits) === bits
end

Base.@kwdef struct PyArrayInterface
    two::Cint = 2
    nd::Cint = 0
    typekind::Cchar = 0
    itemsize::Cint = 0
    flags::Cint = 0
    shape::C.Ptr{Py_intptr_t} = C_NULL
    strides::C.Ptr{Py_intptr_t} = C_NULL
    data::Ptr{Cvoid} = C_NULL
    descr::C.Ptr{PyObject} = C_NULL
end

const G_numpy = Py(UnsafeNew())

function __init_numpy__()
    unsafe_set!(G_numpy, Py_NULLPTR)
    nothing
end

function get_numpy()
    if py_isnull(G_numpy)
        x = G_PyBuiltin.__import__(py_cast(Py, "numpy"))
        PyAPI.Py_IncRef(x)
        unsafe_set!(G_numpy, unsafe_unwrap(x))
    end

    return G_numpy
end

const supported_kinds = Cchar['b', 'i', 'u', 'f', 'c']

const SupportedNDim = 32  # hard coded in numpy
const ShapeType = Union{(Tuple{repeat([Py_intptr_t], i)...} for i = 0:SupportedNDim)...}

function register_root(x::Py, jo::Array)
    ptr = unsafe_unwrap(x)
    PyAPI.Py_IncRef(ptr)
    finalizer(jo) do _
        if RT_is_initialized()
            WITH_GIL(GILNoRaise()) do
                PyAPI.Py_DecRef(ptr)
            end
        end
    end
    jo
end

function _get_capsule_ptr(x::Py)
    u = PyAPI.PyCapsule_GetName(x)
    if u == C_NULL && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    ptr = PyAPI.PyCapsule_GetPointer(x, u)
    if ptr == C_NULL && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    return ptr
end

DevOnly.@staticinclude("CPython.NumPy.DevOnly.jl")
