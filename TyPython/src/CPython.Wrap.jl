import Serialization

# define class ValueBase and RawValue in module jnumpy. maybe other place?
const G_JNUMPY = Py(UnsafeNew())
const valuebasetype = Py(UnsafeNew())

const PyJuliaBase_Type = Ref{C.Ptr{PyObject}}(C_NULL)

const Py_METH_VARARGS = 0x0001 # args are a tuple of arguments
const Py_METH_KEYWORDS = 0x0002  # two arguments: the varargs and the kwargs
const Py_METH_NOARGS = 0x0004  # no arguments (NULL argument pointer)
const Py_METH_O = 0x0008       # single argument (not wrapped in tuple)
const Py_METH_CLASS = 0x0010 # for class methods
const Py_METH_STATIC = 0x0020 # for static methods

# Flags for getting buffers
const PyBUF_SIMPLE = 0x0
const PyBUF_WRITABLE = 0x0001
const PyBUF_WRITEABLE = PyBUF_WRITABLE
const PyBUF_FORMAT = 0x0004
const PyBUF_ND = 0x0008
const PyBUF_STRIDES = (0x0010 | PyBUF_ND)
const PyBUF_C_CONTIGUOUS = (0x0020 | PyBUF_STRIDES)
const PyBUF_F_CONTIGUOUS = (0x0040 | PyBUF_STRIDES)
const PyBUF_ANY_CONTIGUOUS = (0x0080 | PyBUF_STRIDES)
const PyBUF_INDIRECT = (0x0100 | PyBUF_STRIDES)

const PyBUF_CONTIG = (PyBUF_ND | PyBUF_WRITABLE)
const PyBUF_CONTIG_RO = (PyBUF_ND)

const PyBUF_STRIDED = (PyBUF_STRIDES | PyBUF_WRITABLE)
const PyBUF_STRIDED_RO = (PyBUF_STRIDES)

const PyBUF_RECORDS = (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
const PyBUF_RECORDS_RO = (PyBUF_STRIDES | PyBUF_FORMAT)

const PyBUF_FULL = (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
const PyBUF_FULL_RO = (PyBUF_INDIRECT | PyBUF_FORMAT)

const PyBUF_READ = 0x100
const PyBUF_WRITE = 0x200

# the `value` field of `PyJuliaValueObject` indexes into here
const PYJLVALUES = []
# unused indices in PYJLVALUES
const PYJLFREEVALUES = Int[]
const PYJLMETHODS = Vector{Any}()

Py_Type(x::C.Ptr{PyObject}) = C.Ptr{PyObject}(x[].type)
Py_Type(x::Py) = Py_Type(getptr(x))

isflagset(flags, mask) = (flags & mask) == mask

function _pyjl_new(t::C.Ptr{PyObject}, ::C.Ptr{PyObject}, ::C.Ptr{PyObject})
    o = ccall(C.Ptr{PyTypeObject}(t).tp_alloc[], C.Ptr{PyObject}, (C.Ptr{PyObject}, Py_ssize_t), t, 0)
    o == Py_NULLPTR && return Py_NULLPTR
    C.Ptr{PyJuliaValueObject}(o).weaklist[] = Py_NULLPTR
    C.Ptr{PyJuliaValueObject}(o).value[] = 0
    return o
end

function _pyjl_dealloc(o::C.Ptr{PyObject})
    idx = C.Ptr{PyJuliaValueObject}(o).value[]
    if idx != 0
        PYJLVALUES[idx] = nothing
        push!(PYJLFREEVALUES, idx)
    end
    C.Ptr{PyJuliaValueObject}(o).weaklist[] == Py_NULLPTR || PyAPI.PyObject_ClearWeakRefs(o)
    ccall(C.Ptr{PyTypeObject}(Py_Type(o)).tp_free[], Cvoid, (C.Ptr{PyObject},), o)
    nothing
end

function PyJulia_MethodNum(f)
    @nospecialize f
    push!(PYJLMETHODS, f)
    return length(PYJLMETHODS)
end

function pyjl_methodnum(f)
    @nospecialize f
    PyJulia_MethodNum(f)
end

function _pyjl_isnull(o::C.Ptr{PyObject}, ::C.Ptr{PyObject})
    ans = PyJuliaValue_IsNull(o) ? PyAPI._Py_TrueStruct : PyAPI._Py_FalseStruct
    PyAPI.Py_IncRef(ans)
    ans
end

function _pyjl_callmethod(f, self_::C.Ptr{PyObject}, args_::C.Ptr{PyObject}, nargs::Py_ssize_t)
    @nospecialize f
    if PyJuliaValue_IsNull(self_)
        py_seterror!(G_PyBuiltin.TypeError, "Julia object is NULL")
        py_throw()
        return Py_NULLPTR
    end
    in_f = false
    self = PyJuliaValue_GetValue(self_)
    try
        if nargs == 1
            in_f = true
            ans = py_cast(Py, f(self))
            in_f = false
        elseif nargs == 2
            arg1 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 1)) # Borrowed reference here. incref?
            in_f = true
            ans = py_cast(Py, f(self, arg1))
            in_f = false
            # pydel!(arg1)
            # cache?
        elseif nargs == 3
            arg1 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 1))
            arg2 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 2))
            in_f = true
            ans = py_cast(Py, f(self, arg1, arg2))
            in_f = false
        elseif nargs == 4
            arg1 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 1))
            arg2 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 2))
            arg3 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 3))
            in_f = true
            ans = py_cast(Py, f(self, arg1, arg2, arg3))
            in_f = false
        else
            py_seterror!(G_PyBuiltin.TypeError, "__jl_callmethod not implemented for this many arguments")
            py_throw()
        end
        out = getptr(ans)
        PyAPI.Py_IncRef(out)
        return out
    catch exc
        # todo: handle error
        if exc isa PyException
            Base.GC.@preserve exc PyAPI.PyErr_Restore(
                PyAPI.Py_IncRef(getptr(exc.type)),
                PyAPI.Py_IncRef(getptr(exc.value)),
                PyAPI.Py_IncRef(getptr(exc.traceback))
                )
            return Py_NULLPTR
        else
            errmsg = capture_out() do
                Base.showerror(stderr, exc, catch_backtrace())
            end
            py_seterror!(G_JNUMPY.JuliaError, errmsg)
            return Py_NULLPTR
        end
    end
end

function _pyjl_callmethod(o::C.Ptr{PyObject}, args::C.Ptr{PyObject})
    nargs = PyAPI.PyTuple_Size(args)
    @assert nargs > 0
    num = PyAPI.PyLong_AsLongLong(PyAPI.PyTuple_GetItem(args, 0))
    num == -1 && return Py_NULLPTR
    f = PYJLMETHODS[num]
    return _pyjl_callmethod(f, o, args, nargs)::C.Ptr{PyObject}
end

function _pyjl_reduce(self::C.Ptr{PyObject}, ::C.Ptr{PyObject})
    v = _pyjl_serialize(self, Py_NULLPTR)
    v == Py_NULLPTR && return Py_NULLPTR
    args = PyAPI.PyTuple_New(1)
    args == Py_NULLPTR && (PyAPI.Py_DecRef(v); return Py_NULLPTR)
    err = PyAPI.PyTuple_SetItem(args, 0, v)
    err == -1 && (Py_DecRef(args); return Py_NULLPTR)
    red = PyAPI.PyTuple_New(2)
    red == Py_NULLPTR && (PyAPI.Py_DecRef(args); return Py_NULLPTR)
    err = PyAPI.PyTuple_SetItem(red, 1, args)
    err == -1 && (Py_DecRef(red); return Py_NULLPTR)
    f = PyAPI.PyObject_GetAttr(self, py_cast(Py, "_jl_deserialize"))
    f == Py_NULLPTR && (PyAPI.Py_DecRef(red); return Py_NULLPTR)
    err = PyAPI.PyTuple_SetItem(red, 0, f)
    err == -1 && (PyAPI.Py_DecRef(red); return Py_NULLPTR)
    return red
end

function _pyjl_serialize(self::C.Ptr{PyObject}, ::C.Ptr{PyObject})
    try
        io = IOBuffer()
        Serialization.serialize(io, PyJuliaValue_GetValue(self))
        b = take!(io)
        return PyAPI.PyBytes_FromStringAndSize(C.Ptr{Int8}(pointer(b)), sizeof(b))
    catch e
        py_seterror!(G_PyBuiltin.Exception, "error serializing this value")
        rethrow(e)
        return Py_NULLPTR
    end
end

function _pyjl_deserialize(t::C.Ptr{PyObject}, v::C.Ptr{PyObject})
    # todo: error here.
    try
        ptr = Ref{Ptr{Cchar}}()
        len = Ref{Py_ssize_t}()
        err = PyAPI.PyBytes_AsStringAndSize(v, ptr, len)
        err == -1 && return Py_NULLPTR
        io = IOBuffer(unsafe_wrap(Array, Ptr{UInt8}(ptr[]), Int(len[])))
        x = Serialization.deserialize(io)
        return PyJuliaValue_New(t, x)
    catch e
        py_seterror!(G_PyBuiltin.Exception, "error deserializing this value")
        py_throw()
        rethrow(e)
        return Py_NULLPTR
    end
end

PyJuliaValue_IsNull(o::C.Ptr{PyObject}) = C.Ptr{PyJuliaValueObject}(o).value[] == 0

PyJuliaValue_GetValue(o::C.Ptr{PyObject}) = PYJLVALUES[C.Ptr{PyJuliaValueObject}(o).value[]]

PyJuliaValue_SetValue(o::C.Ptr{PyObject}, @nospecialize(v)) = begin
    idx = C.Ptr{PyJuliaValueObject}(o).value[]
    if idx == 0
        if isempty(PYJLFREEVALUES)
            push!(PYJLVALUES, v)
            idx = length(PYJLVALUES)
        else
            idx = pop!(PYJLFREEVALUES)
            PYJLVALUES[idx] = v
        end
        C.Ptr{PyJuliaValueObject}(o).value[] = idx
    else
        PYJLVALUES[idx] = v
    end
    nothing
end

PyJuliaValue_New(t::C.Ptr{PyObject}, @nospecialize(v)) = begin
    if PyAPI.PyType_IsSubtype(t, PyJuliaBase_Type[]) != 1
        py_seterror!(G_PyBuiltin.TypeError, "Expecting a subtype of 'jnumpy.ValueBase'")
        py_throw()
        return Py_NULLPTR
    end
    o = PyAPI.PyObject_CallObject(t, Py_NULLPTR)
    o == Py_NULLPTR && return Py_NULLPTR
    PyJuliaValue_SetValue(o, v)
    return o
end

const _pyjlbase_name = "jnumpy.ValueBase"
const _pyjlbase_type = fill(PyTypeObject())
const _pyjlbase_isnull_name = "_jl_isnull"
const _pyjlbase_callmethod_name = "_jl_callmethod"
const _pyjlbase_reduce_name = "__reduce__"
const _pyjlbase_serialize_name = "_jl_serialize"
const _pyjlbase_deserialize_name = "_jl_deserialize"
const _pyjlbase_methods = Vector{PyMethodDef}()
# const _pyjlbase_as_buffer = fill(PyBufferProcs())
const Py_TPFLAGS_BASETYPE = (0x00000001 << 10)
const Py_TPFLAGS_HAVE_VERSION_TAG = (0x00000001 << 18)

function init_valuebase()
    empty!(_pyjlbase_methods)
    gen_operators()
    append!(_pyjlbase_methods, meths)
    push!(_pyjlbase_methods,
        PyMethodDef(
            ml_name = pointer(_pyjlbase_callmethod_name),
            ml_meth = @cfunction(_pyjl_callmethod, C.Ptr{PyObject}, (C.Ptr{PyObject}, C.Ptr{PyObject})),
            ml_flags = Py_METH_VARARGS,
        ),
        PyMethodDef(
            ml_name = pointer(_pyjlbase_isnull_name),
            ml_meth = @cfunction(_pyjl_isnull, C.Ptr{PyObject}, (C.Ptr{PyObject}, C.Ptr{PyObject})),
            ml_flags = Py_METH_NOARGS,
        ),
        PyMethodDef(
            ml_name = pointer(_pyjlbase_reduce_name),
            ml_meth = @cfunction(_pyjl_reduce, C.Ptr{PyObject}, (C.Ptr{PyObject}, C.Ptr{PyObject})),
            ml_flags = Py_METH_NOARGS,
        ),
        PyMethodDef(
            ml_name = pointer(_pyjlbase_serialize_name),
            ml_meth = @cfunction(_pyjl_serialize, C.Ptr{PyObject}, (C.Ptr{PyObject}, C.Ptr{PyObject})),
            ml_flags = Py_METH_NOARGS,
        ),
        PyMethodDef(
            ml_name = pointer(_pyjlbase_deserialize_name),
            ml_meth = @cfunction(_pyjl_deserialize, C.Ptr{PyObject}, (C.Ptr{PyObject}, C.Ptr{PyObject})),
            ml_flags = Py_METH_O | Py_METH_CLASS,
        ),
        PyMethodDef(),
    )

    _pyjlbase_type[] = PyTypeObject(
            tp_name = pointer(_pyjlbase_name),
            tp_basicsize = sizeof(PyJuliaValueObject),
            tp_new = @cfunction(_pyjl_new, C.Ptr{PyObject}, (C.Ptr{PyObject}, C.Ptr{PyObject}, C.Ptr{PyObject})),
            tp_dealloc = @cfunction(_pyjl_dealloc, Cvoid, (C.Ptr{PyObject},)),
            tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_VERSION_TAG,
            tp_weaklistoffset = fieldoffset(PyJuliaValueObject, 3),
            tp_methods = pointer(_pyjlbase_methods),
            # tp_as_buffer = pointer(_pyjlbase_as_buffer) # todo
        )

    o = PyJuliaBase_Type[] = C.Ptr{PyObject}(pointer(_pyjlbase_type))
    if PyAPI.PyType_Ready(o) == -1
        error("Error initializing 'jnumpy.ValueBase'")
    end
end

function pyisjl(x::Py)
    pyisjl(getptr(x))
end

function pyisjl(x::C.Ptr{PyObject})
    tpx = Py_Type(x)
    PyAPI.PyType_IsSubtype(tpx, valuebasetype) == 1
end

const op_symbol_map = Dict{String, Symbol}(
    "__add__" => :+,
    "__sub__" => :-,
    "__mul__" => :*,
    "__truediv__" => :/,
    "__floordiv__" => :÷,
    "__mod__" => :%,
    "__lshift__" => :<<,
    "__rshift__" => :>>,
    "__and__" => :&,
    "__xor__" => :⊻,
    "__or__" => :|,
    "__eq__" => :(==),
    "__ne__" => :(!=),
    "__le__" => :≤,
    "__lt__" => :<,
    "__ge__" => :≥,
    "__gt__" => :>
)

const meths = Vector{PyMethodDef}()

function gen_cfunc(pyfname::Symbol, op::Symbol)
    quote
        function $pyfname(self_::$C.Ptr{PyObject}, other_::$C.Ptr{PyObject})
            self = $PyJuliaValue_GetValue(self_)
            py_tp = $Py_Type(other_)
            t = $get($PyTypeDict, py_tp, $Py)
            if t !== $Py
                other = $auto_unbox(t, $Py($BorrowReference(), other_))
                try
                    out = $op(self, other)
                    out = $getptr($py_cast($Py, out))
                    $PyAPI.Py_IncRef(out)
                    return out
                catch e
                    # MethodError?
                    # pyexception happends here?
                    if e isa $PyException
                        $CPython.PyAPI.PyErr_SetObject(e.type, e.value)
                    else
                        errmsg = $capture_out() do
                            $Base.showerror($stderr, e, $catch_backtrace())
                        end
                        $py_seterror!($G_JNUMPY.JuliaError, errmsg)
                    end
                    return $Py_NULLPTR
                end
            else
                # return NotImplemented?
                return $getptr(G_PyBuiltin.NotImplemented)
            end
        end
    end
end

function push_meths(pyfname_string_name::String, pyfname::Symbol)
    quote
        push!($meths,
            $PyMethodDef(
                ml_name = $pointer($pyfname_string_name),
                ml_meth = @cfunction($pyfname, $C.Ptr{$PyObject}, ($C.Ptr{$PyObject}, $C.Ptr{$PyObject})),
                ml_flags = $Py_METH_O,
            ),
        )
    end
end

function gen_operators()
    empty!(meths)
    for (k, v) in op_symbol_map
        pyfname = Symbol(k, "##", "_pyfunc")
        eval(gen_cfunc(pyfname, v))
        eval(push_meths(k, pyfname))
    end
end