struct JuliaRaw end
struct PyTuple end
const PyTypeDict = Dict{C.Ptr{PyObject}, Any}()

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
        setindex!(self, auto_unbox(val), auto_unbox(item)...)
    else
        setindex!(self, auto_unbox(val), auto_unbox(item))
    end
    return py_cast(Py, nothing)
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


function auto_unbox_args(pyargs::Py, nargs::Int)
    args = Vector{Any}(undef, nargs)
    for i in 1:nargs
        args[i] = auto_unbox(Py(BorrowReference(), PyAPI.PyTuple_GetItem(pyargs, i-1)))
    end
    return args
end

function auto_unbox_kwargs(pykwargs::Py, nkwargs::Int)
    kwargs = Pair{Symbol, Any}[]
    pyk = PyAPI.PyDict_Keys(pykwargs)
    pyv = PyAPI.PyDict_Values(pykwargs)
    for i in 1:nkwargs
        k = auto_unbox(Py(BorrowReference(), PyAPI.PyList_GetItem(pyk, i-1)))
        v = auto_unbox(Py(BorrowReference(), PyAPI.PyList_GetItem(pyv, i-1)))
        push!(kwargs, attribute_string_to_symbol(k) => v)
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
    n = length(pyarg)
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
    PyTypeDict[unsafe_unwrap(G_jnumpy.JuliaRaw)] = JuliaRaw
end

function _init_jlraw()
    pybuiltins = get_py_builtin()
    pybuiltins.exec(pybuiltins.compile(py_cast(Py,"""
    $("\n"^(@__LINE__()-1))
    class JuliaRaw(JuliaBase):
        __slots__ = ()
        def __getattr__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_getattr)), k)
        def __setattr__(self, k, v):
            self._jl_callmethod($(pyjl_methodnum(pyjlraw_setattr)), k, v)
        def __getitem__(self, k):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_getitem)), k)
        def __setitem__(self, k, v):
            return self._jl_callmethod($(pyjl_methodnum(pyjlraw_setitem)), k, v)
        def __iter__(self):
            pair = self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(iterate))))
            while pair is not None:
                element, state = pair
                yield element
                pair = self._jl_callmethod($(pyjl_methodnum(pyjlraw_op(iterate))), state)

    """), py_cast(Py,@__FILE__()), py_cast(Py, "exec")), G_jnumpy.__dict__)
end

function pyjlraw(v)
    @nospecialize v
    o = Py(PyJuliaValue_New(unsafe_unwrap(G_jnumpy.JuliaRaw), v))
    return o
end

pyjlraw(v::Py) = v

function py_cast(::Type{Py}, o)
    @nospecialize o
    pyjlraw(o)
end

function jl_evaluate(s::String)
    Base.eval(Main, Base.Meta.parseall(s))
end
