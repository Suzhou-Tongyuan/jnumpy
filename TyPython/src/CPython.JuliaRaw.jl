struct JuliaRaw end
const PyTypeDict = Dict{C.Ptr{PyObject}, Any}()
const G_STRING_SYM_MAP = Dict{String, Symbol}()

function attribute_string_to_symbol(x::String)
    get!(G_STRING_SYM_MAP, x) do
        Symbol(x)
    end
end

pyjlraw_repr(self) = py_cast(Py, "<jl $(repr(self))>")
pyjlraw_name(self) = attribute_symbol_to_pyobject(nameof(self))

function pyjlraw_dir(self)
    py_list = G_PyBuiltin.list()
    for k in propertynames(self, true)
        py_list.append(attribute_symbol_to_pyobject(k))
    end
    return py_list
end

function pyjlraw_dir(self::Module)
    ks = Symbol[]
    append!(ks, names(self, all = true, imported = true))
    for m in ccall(:jl_module_usings, Any, (Any,), self)::Vector
        append!(ks, names(m))
    end
    py_list = G_PyBuiltin.list()
    for k in ks
        py_list.append(attribute_symbol_to_pyobject(k))
    end
    return py_list
end

function pyjlraw_getattr(self, k_::Py)
    k = attribute_string_to_symbol(py_coerce(String, k_))
    py_cast(Py, getproperty(self, k))
end

function pyjlraw_setattr(self, k_::Py, v_::Py)
    k = attribute_string_to_symbol(py_coerce(String, k_))
    v = auto_unbox(v_)
    setproperty!(self, k, v)
    py_cast(Py, nothing)
end

function pyjlraw_call(self, pyargs::Py, pykwargs::Py)
    nargs = PyAPI.PyTuple_Size(pyargs)
    nkwargs = PyAPI.PyDict_Size(pykwargs)
    if nkwargs > 0
        args = auto_unbox_tuple(pyargs)
        kwargs = auto_unbox_dict(pykwargs)
        return py_cast(Py, self(args...; kwargs...))
    elseif nargs > 0
        args = auto_unbox_tuple(pyargs)
        return py_cast(Py, self(args...))
    else
        return py_cast(Py, self())
    end
end

function auto_unbox_tuple(pyargs::Py)
    args = Any[]
    py_for(pyargs) do item
        push!(args, auto_unbox(item))
    end
    return args
end

function auto_unbox_dict(pykwargs::Py)
    kwargs = Dict{Symbol, Any}()
    py_for(pykwargs.items()) do item
        pyk = Py(BorrowReference(), PyAPI.PyTuple_GetItem(item, 0))
        pyv = Py(BorrowReference(), PyAPI.PyTuple_GetItem(item, 1))
        k = Symbol(py_coerce(String, pyk))
        kwargs[k] = auto_unbox(pyv)
    end
    return kwargs
end

function auto_unbox(pyarg::Py)
    py_tp = Py_Type(pyarg)
    t = get(PyTypeDict, py_tp, Py)
    auto_unbox(t, pyarg)
end

function auto_unbox(::Type{T}, pyarg::Py) where T
    if T === JuliaRaw
        return PyJuliaValue_GetValue(unsafe_unwrap(pyarg))
    else
        return py_coerce(T, pyarg)
    end
end

function init_typedict()
    pybuiltins = get_py_builtin()
    numpy = get_numpy()
    PyTypeDict[unsafe_unwrap(pybuiltins.None.__class__)] = Nothing
    PyTypeDict[unsafe_unwrap(pybuiltins.bool)] = Bool
    PyTypeDict[unsafe_unwrap(pybuiltins.int)] = Int64
    PyTypeDict[unsafe_unwrap(pybuiltins.float)] = Float64
    PyTypeDict[unsafe_unwrap(pybuiltins.str)] = String
    PyTypeDict[unsafe_unwrap(pybuiltins.complex)] = ComplexF64
    PyTypeDict[unsafe_unwrap(numpy.ndarray)] = AbstractArray
    PyTypeDict[unsafe_unwrap(G_JNUMPY.JuliaRaw)] = JuliaRaw
end

function init_jlraw()
    pybuiltins = get_py_builtin()
    pybuiltins.exec(pybuiltins.compile(py_cast(Py,"""
    $("\n"^(@__LINE__()-1))
    class JuliaRaw(JuliaBase):
        __slots__ = ()
        def __repr__(self):
            if self._jl_isnull():
                return "<jl NULL>"
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_repr)))
        def __getattr__(self, k):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_getattr)), k)
        def __setattr__(self, k, v):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            else:
                self._jl_callmethod($(pyjl_methodnum(pyjlraw_setattr)), k, v)
        def __dir__(self):
            return ValueBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlraw_dir)))
        def __call__(self, *args, **kwargs):
           return self._jl_callmethod($(pyjl_methodnum(pyjlraw_call)), args, kwargs)
        @property
        def __name__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_name)))
    """), py_cast(Py,@__FILE__()), py_cast(Py, "exec")), G_JNUMPY.__dict__)
end

function pyjlraw(v)
    @nospecialize v
    o = Py(PyJuliaValue_New(unsafe_unwrap(G_JNUMPY.JuliaRaw), v))
    return o
end

pyjlraw(v::Py) = v

function py_cast(::Type{Py}, o)
    @nospecialize o
    pyjlraw(o)
end
