import Serialization

# define class JuliaBase and JuliaRaw in module jnumpy.
const G_jnumpy = Py(UnsafeNew())
const PyJuliaBase_Type = Py(UnsafeNew())

function _init_jnumpy()
    jnp = PyAPI.PyImport_ImportModule("jnumpy")
    PyAPI.Py_IncRef(jnp)
    unsafe_set!(G_jnumpy, jnp)
end

const Py_METH_VARARGS = 0x0001 # args are a tuple of arguments
const Py_METH_KEYWORDS = 0x0002  # two arguments: the varargs and the kwargs
const Py_METH_NOARGS = 0x0004  # no arguments (NULL argument pointer)
const Py_METH_O = 0x0008       # single argument (not wrapped in tuple)
const Py_METH_CLASS = 0x0010 # for class methods
const Py_METH_STATIC = 0x0020 # for static methods

# the `value` field of `PyJuliaValueObject` indexes into here
const PYJLVALUES = []
# unused indices in PYJLVALUES
const PYJLFREEVALUES = Int[]
const PYJLMETHODS = Vector{Any}()

Py_Type(x::C.Ptr{PyObject}) = C.Ptr{PyObject}(x[].type)
Py_Type(x::Py) = Py_Type(unsafe_unwrap(x))

function handle_except(e::Exception)
    if e isa PyException
        CPython.PyAPI.PyErr_SetObject(e.type, e.value)
    else
        errmsg = capture_out() do
            Base.showerror(stderr, e, catch_backtrace())
        end
        py_seterror!(_to_py_error(e), errmsg)
    end
    return Py_NULLPTR
end

function handle_except(e::Exception, pyerr::Py)
    if e isa PyException
        CPython.PyAPI.PyErr_SetObject(e.type, e.value)
    else
        errmsg = capture_out() do
            Base.showerror(stderr, e, catch_backtrace())
        end
        py_seterror!(pyerr, errmsg)
    end
    return Py_NULLPTR
end

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
        return Py_NULLPTR
    end
    self = PyJuliaValue_GetValue(self_)
    try
        if nargs == 1
            ans = f(self)::Py
        elseif nargs == 2
            arg1 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 1))
            ans = f(self, arg1)::Py
        elseif nargs == 3
            arg1 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 1))
            arg2 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 2))
            ans = f(self, arg1, arg2)::Py
        elseif nargs == 4
            arg1 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 1))
            arg2 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 2))
            arg3 = Py(BorrowReference(), PyAPI.PyTuple_GetItem(args_, 3))
            ans = f(self, arg1, arg2, arg3)::Py
        else
            py_seterror!(G_PyBuiltin.TypeError, "__jl_callmethod not implemented for this many arguments")
        end
        out = unsafe_unwrap(ans)
        PyAPI.Py_IncRef(out)
        return out
    catch e
        return handle_except(e)
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
    catch
        py_seterror!(G_PyBuiltin.Exception, "error serializing this value")
        return Py_NULLPTR
    end
end

function _pyjl_deserialize(t::C.Ptr{PyObject}, v::C.Ptr{PyObject})
    try
        ptr = Ref{Ptr{Cchar}}()
        len = Ref{Py_ssize_t}()
        err = PyAPI.PyBytes_AsStringAndSize(v, ptr, len)
        err == -1 && return Py_NULLPTR
        io = IOBuffer(unsafe_wrap(Array, Ptr{UInt8}(ptr[]), Int(len[])))
        x = Serialization.deserialize(io)
        return PyJuliaValue_New(t, x)
    catch
        py_seterror!(G_PyBuiltin.Exception, "error deserializing this value")
        return Py_NULLPTR
    end
end

PyJuliaValue_IsNull(o::C.Ptr{PyObject}) = C.Ptr{PyJuliaValueObject}(o).value[] == 0

PyJuliaValue_GetValue(o::C.Ptr{PyObject}) = PYJLVALUES[C.Ptr{PyJuliaValueObject}(o).value[]]

function PyJuliaValue_SetValue(o::C.Ptr{PyObject}, @nospecialize(v))
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

function PyJuliaValue_New(t::C.Ptr{PyObject}, @nospecialize(v))
    if PyAPI.PyType_IsSubtype(t, PyJuliaBase_Type) != 1
        py_seterror!(G_PyBuiltin.TypeError, "Expecting a subtype of 'jnumpy.JuliaBase'")
        return Py_NULLPTR
    end
    o = PyAPI.PyObject_CallObject(t, Py_NULLPTR)
    o == Py_NULLPTR && return Py_NULLPTR
    PyJuliaValue_SetValue(o, v)
    return o
end

function pyjl_repr(self_::C.Ptr{PyObject})
    if PyJuliaValue_IsNull(self_)
        ans = py_cast(Py, "<jl NULL>")
        out = unsafe_unwrap(ans)
        PyAPI.Py_IncRef(out)
        return out
    end
    self = PyJuliaValue_GetValue(self_)
    try
        ans = py_cast(Py, "<jl $(repr(self))>")
        out = unsafe_unwrap(ans)
        PyAPI.Py_IncRef(out)
        return out
    catch e
        handle_except(e)
    end
end

function pyjl_call(self_::C.Ptr{PyObject}, pyargs::C.Ptr{PyObject}, pykwargs::C.Ptr{PyObject})
    if PyJuliaValue_IsNull(self_)
        py_seterror!(G_PyBuiltin.TypeError, "Julia object is NULL")
        return Py_NULLPTR
    end
    self = PyJuliaValue_GetValue(self_)
    nargs = PyAPI.PyTuple_Size(pyargs)
    try
        if pykwargs !== Py_NULLPTR
            nkwargs = PyAPI.PyDict_Size(pykwargs)
            args = auto_unbox_args(Py(BorrowReference(),pyargs), nargs)
            kwargs = auto_unbox_kwargs(Py(BorrowReference(), pykwargs), nkwargs)
            ans = py_cast(Py, self(args...; kwargs...))
        elseif nargs > 0
            args = auto_unbox_args(Py(BorrowReference(), pyargs), nargs)
            ans = py_cast(Py, self(args...))
        else
            ans = py_cast(Py, self())
        end
        out  = unsafe_unwrap(ans)
        PyAPI.Py_IncRef(out)
        return out
    catch e
        handle_except(e)
    end
end

# function pyjl_getattr(self_::C.Ptr{PyObject}, k_::C.Ptr{PyObject})
#     if PyJuliaValue_IsNull(self_)
#         py_seterror!(G_PyBuiltin.TypeError, "Julia object is NULL")
#         return Py_NULLPTR
#     end
#     self = PyJuliaValue_GetValue(self_)
#     try
#         k = attribute_string_to_symbol(py_coerce(String, Py(BorrowReference(), k_)))
#         ans = py_cast(Py, getproperty(self, k))
#         out = unsafe_unwrap(ans)
#         PyAPI.Py_IncRef(out)
#         return out
#     catch e
#         handle_except(e, G_PyBuiltin.AttributeError)
#     end
# end

# function pyjl_setattr(self_::C.Ptr{PyObject}, k_::C.Ptr{PyObject}, v_::C.Ptr{PyObject})
#     if PyJuliaValue_IsNull(self_)
#         py_seterror!(G_PyBuiltin.TypeError, "Julia object is NULL")
#         return Py_NULLPTR
#     end
#     self = PyJuliaValue_GetValue(self_)
#     try
#         k = attribute_string_to_symbol(py_coerce(String, Py(BorrowReference(), k_)))
#         v = auto_unbox(Py(BorrowReference(), v_))
#         setproperty!(self, k, v)
#         out = unsafe_unwrap(PyAPI.Py_None)
#         PyAPI.Py_IncRef(out)
#         return out
#     catch e
#         handle_except(e, G_PyBuiltin.AttributeError)
#     end
# end

const _pyjlbase_name = "jnumpy.JuliaBase"
const _pyjlbase_type = fill(PyTypeObject())
const _pyjlbase_isnull_name = "_jl_isnull"
const _pyjlbase_callmethod_name = "_jl_callmethod"
const _pyjlbase_reduce_name = "__reduce__"
const _pyjlbase_serialize_name = "_jl_serialize"
const _pyjlbase_deserialize_name = "_jl_deserialize"
const _pyjlbase_getattr_name = "_jl_getattr"
const _pyjlbase_setattr_name = "_jl_setattr"
const _pyjlbase_methods = Vector{PyMethodDef}()
const _pyops = Vector{PyMethodDef}()
const Py_TPFLAGS_BASETYPE = (0x00000001 << 10)
const Py_TPFLAGS_HAVE_VERSION_TAG = (0x00000001 << 18)

function _init_juliabase()
    empty!(_pyjlbase_methods)
    generate_operators()
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
            tp_call = @cfunction(pyjl_call, C.Ptr{PyObject}, (C.Ptr{PyObject}, C.Ptr{PyObject}, C.Ptr{PyObject})),
            tp_repr = @cfunction(pyjl_repr, C.Ptr{PyObject}, (C.Ptr{PyObject},)),
        )

    o = C.Ptr{PyObject}(pointer(_pyjlbase_type))
    if PyAPI.PyType_Ready(o) == -1
        error("Error initializing 'jnumpy.JuliaBase'")
    end
    PyAPI.Py_IncRef(o)
    unsafe_set!(PyJuliaBase_Type, o)
end

function pyisjl(x::Py)
    pyisjl(unsafe_unwrap(x))
end

function pyisjl(x::C.Ptr{PyObject})
    tpx = Py_Type(x)
    PyAPI.PyType_IsSubtype(tpx, PyJuliaBase_Type) == 1
end


const op_symbol_map_one_arg = Dict{String, Symbol}(
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

const op_symbol_map_no_arg = Dict{String, Symbol}(
    "__len__" => :length,
    "__neg__" => :-,
    "__pos__" => :+,
    "__invert__" => :~,
    "__abs__" => :abs,
    "__hash__" => :hash,
    "__bool__" => :pyjl_bool,
    "__dir__" => :pyjl_dir
)

function pyjl_bool(self)
    if self isa Number
        return py_cast(Py, o != 0)
    end
    if (self isa AbstractArray || self isa AbstractDict ||
        self isa AbstractSet || self isa AbstractString)
        return py_cast(Py, !isempty(o))
    end
    # return `true` is the default semantics of a Python object
    return py_cast(Py, true)
end

function pyjl_dir(self)
    dir_list = G_PyBuiltin.list()
    for k in propertynames(self, true)
        dir_list.append(attribute_symbol_to_pyobject(k))
    end
    return dir_list
end

function pyjl_dir(self::Module)
    ks = Symbol[]
    append!(ks, names(self, all = true, imported = true))
    for m in ccall(:jl_module_usings, Any, (Any,), self)::Vector
        append!(ks, names(m))
    end
    dir_list = G_PyBuiltin.list()
    for k in ks
        dir_list.append(attribute_symbol_to_pyobject(k))
    end
    return dir_list
end

const meths = Vector{PyMethodDef}()

function generate_cfunc_one_arg(pyfname::Symbol, op::Symbol)
    quote
        function $pyfname(self_::$C.Ptr{$PyObject}, other_::$C.Ptr{$PyObject})
            if $PyJuliaValue_IsNull(self_)
                $py_seterror!($G_PyBuiltin.TypeError, "Julia object is NULL")
                return $Py_NULLPTR
            end
            self = $PyJuliaValue_GetValue(self_)
            py_tp = $Py_Type(other_)
            t = $get($PyTypeDict, py_tp, $Py)
            if t !== $Py
                other = $auto_unbox(t, $Py($BorrowReference(), other_))
                try
                    ans = $op(self, other)
                    out = $unsafe_unwrap($py_cast($Py, ans))
                    $PyAPI.Py_IncRef(out)
                    return out
                catch e
                    $handle_except(e)
                end
            else
                out = $unsafe_unwrap($G_PyBuiltin.NotImplemented)
                PyAPI.Py_IncRef(out)
                return out
            end
        end
    end
end

function generate_cfunc_no_arg(pyfname::Symbol, op::Symbol)
    quote
        function $pyfname(self_::$C.Ptr{$PyObject}, ::$C.Ptr{$PyObject})
            if $PyJuliaValue_IsNull(self_)
                $py_seterror!($G_PyBuiltin.TypeError, "Julia object is NULL")
                return $Py_NULLPTR
            end
            self = $PyJuliaValue_GetValue(self_)
            try
                ans = $op(self)
                out = $unsafe_unwrap($py_cast($Py, ans))
                $PyAPI.Py_IncRef(out)
                return out
            catch e
                $handle_except(e)
            end
        end
    end
end

function push_meths_one_arg(pyfname_string_name::String, pyfname::Symbol)
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

function push_meths_no_arg(pyfname_string_name::String, pyfname::Symbol)
    quote
        push!($meths,
            $PyMethodDef(
                ml_name = $pointer($pyfname_string_name),
                ml_meth = @cfunction($pyfname, $C.Ptr{$PyObject}, ($C.Ptr{$PyObject}, $C.Ptr{$PyObject})),
                ml_flags = $Py_METH_NOARGS,
            ),
        )
    end
end


function generate_operators()
    empty!(meths)
    for (k, v) in op_symbol_map_one_arg
        pyfname = Symbol(k, "##", "_pyfunc")
        eval(generate_cfunc_one_arg(pyfname, v))
        eval(push_meths_one_arg(k, pyfname))
    end
    for (k, v) in op_symbol_map_no_arg
        pyfname = Symbol(k, "##", "_pyfunc")
        eval(generate_cfunc_no_arg(pyfname, v))
        eval(push_meths_no_arg(k, pyfname))
    end
end
