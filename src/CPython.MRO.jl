function attribute_symbol_to_pyobject(x::Symbol)
    get!(G_ATTR_SYM_MAP, x) do
        Py(PyAPI.PyUnicode_FromString(string(x)))
    end
end

@noinline function Base.getproperty(self::Py, name::Symbol)::Py
    Py(PyAPI.PyObject_GetAttr(self, attribute_symbol_to_pyobject(name)))
end

@noinline function Base.setproperty!(self::Py, name::Symbol, value::Py)::Py
    PyAPI.PyObject_SetAttr(self, attribute_symbol_to_pyobject(name), value)
    value
end

"""
    py_tuple_create(py::Py...)

Create a Python tuple from variadic arguments.
"""
function py_tuple_create(x::Vararg{Py, N})::Py where N
    argtuple = PyAPI.PyTuple_New(N)
    unroll_do!(Val(N), argtuple, args) do i, argtuple, args
        PyAPI.Py_IncRef(args[i])
        PyAPI.PyTuple_SetItem(argtuple, i-1, args[i])
    end
    return Py(argtuple)
end

@noinline function Base.getindex(self::Py, ind::Py, inds::Py...)::Py
    isempty(inds) && return Py(PyAPI.PyObject_GetItem(self, ind))
    return Py(PyAPI.PyObject_GetItem(self, py_tuple_create(ind, inds...)))
end

@noinline function Base.setindex!(self::Py, value::Py, ind, inds::Py...)::Py
    if isempty(inds)
        PyAPI.PyObject_SetItem(self, value, ind)
    else
        PyAPI.PyObject_SetItem(self, py_tuple_create(ind, inds...))
    end
    value
end

function (py::Py)(args::Vararg{Py, N}; kwargs...) where N
    if isempty(kwargs)
        if isempty(args)
            return Py(PyAPI.PyObject_CallObject(py, Py_NULLPTR))
        else
            argtuple = PyAPI.PyTuple_New(N)
            try
                unroll_do!(Val(N), argtuple, args) do i, argtuple, args
                    PyAPI.Py_IncRef(args[i])
                    PyAPI.PyTuple_SetItem(argtuple, i-1, args[i])
                end
                return Py(PyAPI.PyObject_Call(py, argtuple, Py_NULLPTR))
            finally
                PyAPI.Py_DecRef(argtuple)
            end
        end
    else
        if eltype(kwargs) !== Pair{Symbol, Py}
            error("kwargs must be Py objects")
        end
        argtuple = PyAPI.PyTuple_New(length(args))
        argdict = PyAPI.PyDict_New()
        try
            unroll_do!(Val(N), argtuple, args) do i, argtuple, args
                PyAPI.Py_IncRef(argtuple[i])
                PyAPI.PyTuple_SetItem(argtuple, i-1, args[i])
            end
            for (key::Symbol, arg::Py) in kwargs
                PyAPI.PyDict_SetItem(argdict, attribute_symbol_to_pyobject(key), arg)
            end
            return Py(PyAPI.PyObject_Call(py, argtuple, argdict))
        finally
            PyAPI.Py_DecRef(argtuple)
            PyAPI.Py_DecRef(argdict)
        end
    end
end

function Base.length(self::Py)
    PyAPI.PyObject_Length(self)
end

function py_eq(x::Py, y::Py)
    Py(PyAPI.PyObject_RichCompare(x, y, Py_EQ))
end

function py_ne(x::Py, y::Py)
    Py(PyAPI.PyObject_RichCompare(x, y, Py_NE))
end

function py_lt(x::Py, y::Py)
    Py(PyAPI.PyObject_RichCompare(x, y, Py_LT))
end

function py_le(x::Py, y::Py)
    Py(PyAPI.PyObject_RichCompare(x, y, Py_LE))
end

function py_gt(x::Py, y::Py)
    Py(PyAPI.PyObject_RichCompare(x, y, Py_GT))
end

function py_ge(x::Py, y::Py)
    Py(PyAPI.PyObject_RichCompare(x, y, Py_GE))
end

@eval function Base.$(:(==))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_eq(x, y)) != 0
end

@eval function Base.$(:(!=))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_ne(x, y)) != 0
end

@eval function Base.$(:(>))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_lt(x, y)) != 0
end

@eval function Base.$(:(>=))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_le(x, y)) != 0
end

@eval function Base.$(:(<))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_gt(x, y)) != 0
end

@eval function Base.$(:(<=))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_ge(x, y)) != 0
end

function py_dir(x::Py)
    Py(PyAPI.PyObject_Dir(x))
end

"""
    py_coerce(t, py::Py)
Perform no cast but use the underlying type for coercions.
"""
py_coerce(t, py::Py)

function py_coerce(::Type{T}, py::Py) where T <: Integer
    i = PyAPI.PyLong_AsLongLong(py)
    if i == -1 && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    convert(T, i)
end

function py_coerce(::Type{T}, py::Py) where T <: AbstractFloat
    d = PyAPI.PyFloat_AsDouble(py)
    if d == -1.0 && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    convert(T, d)
end

function py_coerce(::Type{T}, py::Py) where T <: Complex
    d = PyAPI.PyComplex_AsCComplex(py) :: Py_complex
    if d.real == -1.0 && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    convert(T, complex(d.real, d.imag))
end
