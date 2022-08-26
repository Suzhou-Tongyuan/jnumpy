pyjlraw_repr(self) = py_cast(Py, "<jl $(repr(self))>")

function pyjlraw_getattr(self, k_::Py)
    k = Symbol(py_coerce(String, k_))
    # convertion?
    py_cast(Py, getproperty(self, k))
end

function pyjlraw_call(self, pyargs::Py, pykwargs::Py)
    # todo
    # unbox pyargs and pykwargs
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
    if haskey(PyTypeDict, py_tp)
        t = PyTypeDict[py_tp]
        if t === JLRawValue
            return PyJuliaValue_GetValue(getptr(pyarg))
        else
            return py_coerce(t, pyarg)
        end
    else
        return pyarg
    end
end

const PyTypeDict = Dict{C.Ptr{PyObject}, Any}()

function init_typedict()
    pybuiltins = get_py_builtin()
    numpy = get_numpy()
    # subclass?
    PyTypeDict[getptr(pybuiltins.None.__class__)] = Nothing
    PyTypeDict[getptr(pybuiltins.bool)] = Bool
    PyTypeDict[getptr(pybuiltins.int)] = Int64
    PyTypeDict[getptr(pybuiltins.float)] = Float64
    PyTypeDict[getptr(pybuiltins.str)] = String
    PyTypeDict[getptr(pybuiltins.complex)] = ComplexF64
    PyTypeDict[getptr(numpy.ndarray)] = AbstractArray
    PyTypeDict[getptr(G_JNUMPY.RawValue)] = JLRawValue
end

struct JLRawValue end

function init_jlwrap_raw()
    pybuiltins = get_py_builtin()
    pybuiltins.exec(pybuiltins.compile(py_cast(Py,"""
    $("\n"^(@__LINE__()-1))
    class RawValue(ValueBase):
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
        def __call__(self, *args, **kwargs):
           return self._jl_callmethod($(pyjl_methodnum(pyjlraw_call)), args, kwargs)
    """), py_cast(Py,@__FILE__()), py_cast(Py, "exec")), G_JNUMPY.__dict__)
end

function pyjlraw(v)
    @nospecialize v
    o = Py(PyJuliaValue_New(getptr(G_JNUMPY.RawValue), v))
    return o
end

pyjlraw(v::Py) = v

function py_cast(::Type{Py}, o)
    @nospecialize o
    pyjlraw(o)
end