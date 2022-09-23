struct JuliaRaw end
struct PyTuple end
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
    dir_list = G_PyBuiltin.list()
    for k in propertynames(self, true)
        dir_list.append(attribute_symbol_to_pyobject(k))
    end
    return dir_list
end

function pyjlraw_dir(self::Module)
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

function pyjlraw_getitem(self, item::Py)::Py
    py_tp = Py_Type(item)
    t = get(PyTypeDict, py_tp, Py)
    if t <: PyTuple
        py_cast(Py, getindex(self, auto_unbox(item)...))
    else
        py_cast(Py, getindex(self, auto_unbox(item)))
    end
end

function pyjlraw_setitem(self, item::Py, val::Py)::Py
    py_tp = Py_Type(item)
    t = get(PyTypeDict, py_tp, Py)
    if t <: PyTuple
        setindex!(
            self,
            auto_unbox(val),
            auto_unbox(item)...)
    else
        setindex!(
            self,
            auto_unbox(val),
            auto_unbox(item))
    end
    return py_cast(Py, nothing)
end

function pyjlraw_bool(self)
    if self isa Number
        return py_cast(Bool, o != 0)
    end
    if (self isa AbstractArray || self isa AbstractDict ||
        self isa AbstractSet || self isa AbstractString)
        return py_cast(Bool, !isempty(o))
    end
    # return `true` is the default semantics of a Python object
    return py_cast(Bool, true)
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

struct pyjlraw_revop{OP}
    op :: OP
end

(op::pyjlraw_revop)(self) = py_cast(Py, op.op(self))

function (op::pyjlraw_revop)(self, other_::Py)
    py_tp = Py_Type(other_)
    t = get(PyTypeDict, py_tp, Py)
    if t !== Py
        other = auto_unbox(t, other_)
        return py_cast(Py, op.op(other, self))
    else
        return G_PyBuiltin.NotImplemented
    end
end

function (op::pyjlraw_revop)(self, other_::Py, other2_::Py)
    t1 = get(PyTypeDict, Py_Type(other_), Py)
    t2 = get(PyTypeDict, Py_Type(other2_), Py)
    if t1 !== Py && t2 !== Py
        other = auto_unbox(t, other_)
        other2 = auto_unbox(t, other2_)
        return py_cast(Py, op.op(other, self, other2))
    else
        return G_PyBuiltin.NotImplemented
    end
end

function pyjlraw_call(self, pyargs::Py, pykwargs::Py)
    nargs = PyAPI.PyTuple_Size(pyargs)
    nkwargs = PyAPI.PyDict_Size(pykwargs)
    if nkwargs > 0
        args = auto_unbox_args(pyargs)
        kwargs = auto_unbox_kwargs(pykwargs)
        return py_cast(Py, self(args...; kwargs...))
    elseif nargs > 0
        args = auto_unbox_args(pyargs)
        return py_cast(Py, self(args...))
    else
        return py_cast(Py, self())
    end
end

function auto_unbox_args(pyargs::Py)
    args = Any[]
    py_for(pyargs) do item
        push!(args, auto_unbox(item))
    end
    return args
end

function auto_unbox_kwargs(pykwargs::Py)
    kwargs = Pair{Symbol, Any}[]
    py_for(pykwargs.items()) do item
        pyk = Py(BorrowReference(), PyAPI.PyTuple_GetItem(item, 0))
        pyv = Py(BorrowReference(), PyAPI.PyTuple_GetItem(item, 1))
        k = Symbol(py_coerce(String, pyk))
        push!(kwargs, k => auto_unbox(pyv))
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

function auto_unbox(::Type{PyTuple}, pyarg::Py)
    n = length(py)
    return Tuple(auto_unbox(pyarg[py_cast(Py, i-1)]) for i in 1:n)
end

function _init_typedict()
    pybuiltins = get_py_builtin()
    numpy = get_numpy()
    PyTypeDict[unsafe_unwrap(pybuiltins.None.__class__)] = Nothing
    PyTypeDict[unsafe_unwrap(pybuiltins.tuple)] = PyTuple
    PyTypeDict[unsafe_unwrap(pybuiltins.bool)] = Bool
    PyTypeDict[unsafe_unwrap(pybuiltins.int)] = Int64
    PyTypeDict[unsafe_unwrap(pybuiltins.float)] = Float64
    PyTypeDict[unsafe_unwrap(pybuiltins.str)] = String
    PyTypeDict[unsafe_unwrap(pybuiltins.complex)] = ComplexF64
    PyTypeDict[unsafe_unwrap(numpy.ndarray)] = AbstractArray
    PyTypeDict[unsafe_unwrap(G_JNUMPY.JuliaRaw)] = JuliaRaw
end

function _init_jlraw()
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
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_getattr)), k)
        def __setattr__(self, k, v):
            self._jl_callmethod($(pyjl_methodnum(pyjlraw_setattr)), k, v)
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_getitem)), k)
        def __setitem__(self, k, v):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_setitem)), k, v)
        def __dir__(self):
            return JuliaBase.__dir__(self) + self._jl_callmethod($(pyjl_methodnum(pyjlraw_dir)))
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
        def __radd__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(+))), other)
        def __rsub__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(-))), other)
        def __rmul__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(*))), other)
        def __rtruediv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(/))), other)
        def __rfloordiv__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(÷))), other)
        def __rmod__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(%))), other)
        def __rpow__(self, other, modulo=None):
            if modulo is None:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(^))), other)
            else:
                return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(powermod))), other, modulo)
        def __rlshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(<<))), other)
        def __rrshift__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(>>))), other)
        def __rand__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(&))), other)
        def __rxor__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(⊻))), other)
        def __ror__(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_revop(|))), other)
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
        def __bool__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_bool)))
        def __iter__(self):
            pair = self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(iterate))))
            while pair is not None:
                element, state = pair
                yield element
                pair = self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(iterate))), state)
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
