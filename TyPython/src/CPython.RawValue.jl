pyjlraw_repr(self) = py_cast(Py, "<jl $(repr(self))>")
pyjlraw_name(self) = py_cast(Py, string(nameof(self)))

function pyjlraw_dir(self)
    py_list = G_PyBuiltin.list()
    for k in propertynames(self, true)
        py_list.append(attribute_symbol_to_pyobject(x))
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
    for x in ks
        py_list.append(attribute_symbol_to_pyobject(x))
    end
    return py_list
end

const G_STRING_SYM_MAP = Dict{String, Symbol}()

function attribute_string_to_symbol(x::String)
    get!(G_STRING_SYM_MAP, x) do
        Symbol(x)
    end
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

struct pyjlraw_op{OP}
    op :: OP
end

(op::pyjlraw_op)(self) = py_cast(Py, op.op(self))

function (op::pyjlraw_op)(self, other_::Py)
    py_tp = Py_Type(other_)
    t = get(PyTypeDict, py_tp, Py)
    if t !== Py
        other = auto_unbox(t, other_)
        return py_cast(Py, op.op(self, other))
    else
        return G_PyBuiltin.NotImplemented
    end
end

function (op::pyjlraw_op)(self, other_::Py, other2_::Py)
    t1 = get(PyTypeDict, Py_Type(other_), Py)
    t2 = get(PyTypeDict, Py_Type(other2_), Py)
    if t1 !== Py && t2 !== Py
        other = auto_unbox(t, other_)
        other2 = auto_unbox(t, other2_)
        return py_cast(Py, op.op(self, other, other2))
    else
        return G_PyBuiltin.NotImplemented
    end
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
    t = get(PyTypeDict, py_tp, Py)
    auto_unbox(t, pyarg)
end

function auto_unbox(::Type{T}, pyarg::Py) where T
    if T === JLRawValue
        return PyJuliaValue_GetValue(getptr(pyarg))
    else
        return py_coerce(T, pyarg)
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
        def __setattr__(self, k, v):
            if k.startswith("__") and k.endswith("__"):
                raise AttributeError(k)
            else:
                self._jl_callmethod($(pyjl_methodnum(pyjlraw_setattr)), k, v)
        def __dir__(self):
            return ValueBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlraw_dir)))
        def __call__(self, *args, **kwargs):
           return self._jl_callmethod($(pyjl_methodnum(pyjlraw_call)), args, kwargs)
        def __len__(self):
           return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(length))))
        def __pos__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(+))))
        def __neg__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(-))))
        def __abs__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(abs))))
        def __invert__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(~))))
        def __add__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(+))), other)
        def __sub__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(-))), other)
        def __mul__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(*))), other)
        def __truediv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(/))), other)
        def __floordiv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(÷))), other)
        def __mod__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(%))), other)
        def __pow__(self, other, modulo=None):
            if modulo is None:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(^))), other)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(powermod))), other, modulo)
        def __lshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(<<))), other)
        def __rshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(>>))), other)
        def __and__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(&))), other)
        def __xor__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(⊻))), other)
        def __or__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(|))), other)
        def __eq__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(==))), other)
        def __ne__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(!=))), other)
        def __le__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(≤))), other)
        def __lt__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(<))), other)
        def __ge__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(≥))), other)
        def __gt__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(>))), other)
        def __hash__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(hash))))
        @property
        def __name__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_name)))
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