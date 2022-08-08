struct CPyFunction{Params<:Tuple, Return}
    ptr :: Ptr{Cvoid}
end

# function Base.show(io::IO, ::Type{CPyFunction{Params, Return}}) where {Params, Return}
#     if Params isa DataType && Return isa DataType
#         print(io, "PyC(" * join(collect(Params.parameters), ", ") * ")::" * string(Return))
#     else
#         print(io, "CPyFunction{$(Params), $(Return)}")
#     end
# end

struct Except{Return, ErrorCode} end
struct NoErrorCode end
Except(@nospecialize(errcode), rety::DataType) = Except{rety,  convert(rety, errcode)}

Base.convert(::Type{CFunc}, ptr::Ptr{Cvoid}) where CFunc <: CPyFunction = CFunc(ptr)
function cfunc_t(@nospecialize(args::DataType...))
    isempty(args) && cfunc_t(Cvoid)
    return CPyFunction{Tuple{args[1:end-1]...}, args[end]}
end

unsafe_broaden_type(::Type{Cstring}) = Union{Cstring, AbstractString}
unsafe_broaden_type(::Type{Cchar}) = Union{Char, UInt8}
unsafe_broaden_type(::Union{Type{Cfloat}, Type{Cdouble}}) = AbstractFloat
unsafe_broaden_type(::Union{Type{Csize_t}, Type{Cshort}, Type{Cint}}) = Integer
unsafe_broaden_type(::Union{Type{Cuint}, Type{Cushort}, Type{Culonglong}, Type{Culong}}) = Unsigned
unsafe_broaden_type(::Type{Ptr{O}}) where O = isbitstype(O) ? Union{Base.RefValue{O}, Ptr{O}} : Ptr{O}
unsafe_broaden_type(::Type{C.Ptr{O}}) where O =
    O === PyObject ?
        Union{C.Ptr{PyObject}, Py} :
    isbitstype(O) ?
        Union{C.Ptr{O}, Base.RefValue{O}} :
        C.Ptr{O}
unsafe_broaden_type(t) = t

"""
    unsafe_broaden_type(t)

Given an underlying type, return a type (can be union) that is more high-level but convertable to the underlying type.

i.e., given `t::DataType`, `T <: unsafe_broaden_type(t)` and `o :: T`,
we have
```
Base.unsafe_convert(t, Base.cconvert(t), o) isa t
```
"""
unsafe_broaden_type(t::DataType)

const _support_ccall_sigs = Set{DataType}()

function _support_ccall!(ln::LineNumberNode, @nospecialize(CFuncType::DataType))
    CFuncType <: CPyFunction || error("_support_ccall! expects a CPyFunction type")
    (Params, Return) = CFuncType.parameters
    ErrorCode = NoErrorCode()
    if Return <: Except
        (Return, ErrorCode) = Return.parameters
    end
    CFuncType in _support_ccall_sigs && return :nothing
    push!(_support_ccall_sigs, CFuncType)
    narg = length(Params.parameters)
    parTypes = Tuple(Params.parameters)
    parNames = [ gensym("arg$i") for i = 1:narg ]
    unsafeNames = [ gensym("unsafe_arg$i") for i = 1:narg ]
    parAnns = [:($(parNames[i])::$(unsafe_broaden_type(parTypes[i]))) for i = 1:narg ]
    lowering = Expr(:block, ln, ln)
    for i in 1:narg
        push!(lowering.args, :(local $(unsafeNames[i]) = Base.cconvert($(parTypes[i]), $(parNames[i]))))
    end
    expr_ccall = :(ccall(cfunc.ptr, $Return, $(Expr(:tuple, parTypes...)), $(unsafeNames...)))
    gc_call = Expr(:gc_preserve, Expr(:block, expr_ccall), unsafeNames...)
    sym_retval = gensym("return")
    push!(lowering.args, :(local $sym_retval = $gc_call))
    check_call = if ErrorCode isa NoErrorCode
        :(return $sym_retval)
    else
        Expr(:if,
            :($ErrorCode === $sym_retval),
            :($CPython.py_throw()),
            :(return $sym_retval)
        )
    end
    push!(lowering.args, check_call)
    Expr(:function, :((cfunc::$CFuncType)($(parAnns...))), lowering)
end

const PyCFunction = Ptr{Cvoid}  # (PyO*, PyO*) -> PyO*
const PyCFunctionFast = Ptr{Cvoid} # (PyO*, const PyO*, Py_ssize_t) -> PyO*
const PyCFunctionWithKeywords = Ptr{Cvoid} # (PyO*, PyO*, Py_ssize_t, PyO* kwnames) -> PyO*
const METH_VARARGS = Cint(0x0001)
const METH_KEYWORDS = Cint(0x0002)
const METH_NOARGS = Cint(0x0004)
const METH_FASTCALL = Cint(0x0080)  # since 2016 / Python 3.6
const Py_ssize_t = Cssize_t
const Py_hash_t = Cssize_t

Base.@kwdef struct PyMethodDef
    ml_name  :: Cstring = C_NULL
    ml_meth  :: PyCFunction = C_NULL
    ml_flags :: Cint = 0
    ml_doc   :: Cstring = C_NULL
end

Base.@kwdef struct PyObject
    # assumes _PyObject_HEAD_EXTRA is empty
    refcnt::Py_ssize_t = 0
    type::Ptr{Cvoid} = C_NULL # really is Ptr{PyObject} or Ptr{PyTypeObject} but Julia 1.3 and below get the layout incorrect when circular types are involved
end

Base.@kwdef struct Py_buffer
    buf::Ptr{Cvoid} = C_NULL
    obj::Ptr{Cvoid} = C_NULL
    len::Py_ssize_t = 0
    itemsize::Py_ssize_t = 0
    readonly::Cint = 0
    ndim::Cint = 0
    format::Cstring = C_NULL
    shape::Ptr{Py_ssize_t} = C_NULL
    strides::Ptr{Py_ssize_t} = C_NULL
    suboffsets::Ptr{Py_ssize_t} = C_NULL
    internal::Ptr{Cvoid} = C_NULL
end

Base.@kwdef struct PyBufferProcs
    get::Ptr{Cvoid} = C_NULL # (o, Ptr{Py_buffer}, Cint) -> Cint
    release::Ptr{Cvoid} = C_NULL # (o, Ptr{Py_buffer}) -> Cvoid
end

Base.@kwdef struct PyTypeObject
    ob_base :: PyObject
    ob_size :: Py_ssize_t
    tp_name::Cstring = C_NULL
    tp_basicsize::Py_ssize_t = 0
    tp_itemsize::Py_ssize_t = 0
    tp_dealloc::Ptr{Cvoid} = C_NULL
    tp_vectorcall_offset::Py_ssize_t = 0
    tp_getattr::Ptr{Cvoid} = C_NULL
    tp_setattr::Ptr{Cvoid} = C_NULL
    tp_as_async::Ptr{Cvoid} = C_NULL
    tp_repr::Ptr{Cvoid} = C_NULL
    # Method suites for standard classes
    tp_as_number::Ptr{Cvoid} = C_NULL
    tp_as_sequence::Ptr{Cvoid} = C_NULL
    tp_as_mapping::Ptr{Cvoid} = C_NULL
    # More standard operations (here for binary compatibility)
    tp_hash::Ptr{Cvoid} = C_NULL
    tp_call::Ptr{Cvoid} = C_NULL
    tp_str::Ptr{Cvoid} = C_NULL
    tp_getattro::Ptr{Cvoid} = C_NULL
    tp_setattro::Ptr{Cvoid} = C_NULL
    # Functions to access object as input/output buffer
    tp_as_buffer::Ptr{Cvoid} = C_NULL
    # Flags to define presence of optional/expanded feature
    tp_flags::Culong = 0
    tp_doc::Cstring = C_NULL
    tp_traverse::Ptr{Cvoid} = C_NULL
    tp_clear::Ptr{Cvoid} = C_NULL
    tp_richcompare::Ptr{Cvoid} = C_NULL
    tp_weaklistoffset::Py_ssize_t = 0
    tp_iter::Ptr{Cvoid} = C_NULL
    tp_iternext::Ptr{Cvoid} = C_NULL
    tp_methods::Ptr{PyMethodDef} = C_NULL
    tp_members::Ptr{Cvoid} = C_NULL
    tp_getset::Ptr{Cvoid} = C_NULL
    tp_base::Ptr{Cvoid} = C_NULL
    tp_dict::Ptr{Cvoid} = C_NULL
    tp_descr_get::Ptr{Cvoid} = C_NULL
    tp_descr_set::Ptr{Cvoid} = C_NULL
    tp_dictoffset::Py_ssize_t = 0
    tp_init::Ptr{Cvoid} = C_NULL
    tp_alloc::Ptr{Cvoid} = C_NULL
    tp_new::Ptr{Cvoid} = C_NULL
    tp_free::Ptr{Cvoid} = C_NULL
    tp_is_gc::Ptr{Cvoid} = C_NULL
    tp_bases::Ptr{Cvoid} = C_NULL
    tp_mro::Ptr{Cvoid} = C_NULL
    tp_cache::Ptr{Cvoid} = C_NULL
    tp_subclasses::Ptr{Cvoid} = C_NULL
    tp_weaklist::Ptr{Cvoid} = C_NULL
    tp_del::Ptr{Cvoid} = C_NULL

    # Type attribute cache version tag. Added in version 2.6
    tp_version_tag::Cuint = 0

    tp_finalize::Ptr{Cvoid} = C_NULL
    tp_vectorcall::Ptr{Cvoid} = C_NULL
end

struct Py_complex
    real::Cdouble
    imag::Cdouble
end

function Base.convert(::Type{Py_complex}, o::Complex)
    real = convert(Cdouble, o.re)
    imag = convert(Cdouble, o.im)
    return Py_complex(real, imag)
end

const SIZE_PyGILState = sizeof(Cint) * 8
primitive type PyGILState SIZE_PyGILState end

const Py_NULLPTR = C.Ptr{PyObject}(C_NULL)

const Py_LT = Cint(0)
const Py_LE = Cint(1)
const Py_EQ = Cint(2)
const Py_NE = Cint(3)
const Py_GT = Cint(4)
const Py_GE = Cint(5)
