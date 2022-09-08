function attribute_symbol_to_pyobject(x::Symbol)
    get!(G_ATTR_SYM_MAP, x) do
        Py(PyAPI.PyUnicode_FromString(string(x)))
    end
end

@noinline function Base.getproperty(self::Py, name::Symbol)::Py
    Py(PyAPI.PyObject_GetAttr(self, attribute_symbol_to_pyobject(name)))
end

@noinline function Base.hasproperty(self::Py, name::Symbol)::Bool
   PyAPI.PyObject_HasAttr(self, attribute_symbol_to_pyobject(name)) != 0
end

@noinline function Base.setproperty!(self::Py, name::Symbol, value::Py)::Py
    PyAPI.PyObject_SetAttr(self, attribute_symbol_to_pyobject(name), value)
    value
end

"""
    py_tuple_create(py::Py...)

Create a Python tuple from variadic arguments.
"""
function py_tuple_create(args::Vararg{Py, N})::Py where N
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

@noinline function Base.setindex!(self::Py, value::Py, ind::Py, inds::Py...)::Py
    if isempty(inds)
        PyAPI.PyObject_SetItem(self, ind, value)
    else
        PyAPI.PyObject_SetItem(self, py_tuple_create(ind, inds...), value)
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
                PyAPI.Py_IncRef(args[i])
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
    return PyAPI.PyObject_IsTrue(py_gt(x, y)) != 0
end

@eval function Base.$(:(>=))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_ge(x, y)) != 0
end

@eval function Base.$(:(<))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_lt(x, y)) != 0
end

@eval function Base.$(:(<=))(x::Py, y::Py)
    return PyAPI.PyObject_IsTrue(py_le(x, y)) != 0
end

function py_dir(x::Py)
    Py(PyAPI.PyObject_Dir(x))
end

"""
    py_coerce(t, py::Py)

Perform no cast but use the underlying type for coercions.
"""
py_coerce(t, py::Py)

py_coerce(::Type{Py}, py::Py) = py

function py_coerce(::Type{Nothing}, py::Py)
    py_equal_identity(py, PyAPI.Py_None) && return nothing
    py_seterror!(G_PyBuiltin.TypeError, "expected None")
    py_throw()
end

function py_coerce(::Type{Bool}, py::Py)
    py_equal_identity(py, PyAPI.Py_True) && return true
    py_equal_identity(py, PyAPI.Py_False) && return false
    py_seterror!(G_PyBuiltin.TypeError, "expected bool type")
    py_throw()
end

function py_coerce(::Type{T}, py::Py)::T where T <: Integer
    i = PyAPI.PyLong_AsSsize_t(py)
    if i == -1 && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    convert(T, i)
end

function py_coerce(::Type{T}, py::Py)::T where T <: AbstractFloat
    if PyAPI.PyObject_IsInstance(py, PyAPI.PyFloat_Type) == 0
        py_seterror!(G_PyBuiltin.TypeError, "expected float type")
        py_throw()
    end
    py_cast(T, py)
end

function py_coerce(::Type{T}, py::Py)::T where T <: Complex
    if PyAPI.PyObject_IsInstance(py, PyAPI.PyComplex_Type) == 0
        py_seterror!(G_PyBuiltin.TypeError, "expected complex type")
        py_throw()
    end
    py_cast(T, py)
end

function py_coerce(::Type{T}, py::Py)::T where T <: AbstractString
    size_ref = Ref(0)
    buf = PyAPI.PyUnicode_AsUTF8AndSize(py , size_ref)
    return convert(T, Base.unsafe_string(buf, size_ref[]))
end

function py_coerce(::Type{TArray}, py::Py)::TArray where {TArray <: AbstractArray}
    np = get_numpy()
    if PyAPI.PyObject_IsInstance(py, np.ndarray) == 0
        py_seterror!(G_PyBuiltin.TypeError, "expected numpy array")
        py_throw()
    end
    convert(TArray, from_ndarray(py))
end

function unsafe_pytuple_get(o::Py, i::Integer)
    item = PyAPI.PyTuple_GetItem(o, i)
    PyAPI.Py_IncRef(item)
    return Py(item)
end

function check_tuple_length(py::Py, len_tuple::Int)
    len_py_tuple = PyAPI.PyTuple_Size(py)
    if len_tuple != len_py_tuple
        error("length of $py is $len_py_tuple, but expect $len_tuple")
    end
end

@generated function py_coerce(::Type{T}, py::Py)::T where T <: Tuple
    T === Tuple && return :(error("$T should not equal to Tuple"))
    T isa DataType || return :(error("$T is not a data type"))
    len_tuple = length(T.parameters)
    result = Expr(:tuple, [:($py_coerce($t, $unsafe_pytuple_get(py, $(i - 1)))) for (i, t) in enumerate(T.parameters)]...)
    quote
        if $CPython.PyAPI.PyObject_IsInstance(py,  $CPython.PyAPI.PyTuple_Type) == 0
            $py_seterror!(G_PyBuiltin.TypeError, "expected a tuple")
            $py_throw()
        end
        $check_tuple_length(py, $len_tuple)
        $result
    end
end

function py_equal_identity(x::Union{Py, C.Ptr{PyObject}}, y::Union{Py, C.Ptr{PyObject}})
    unsafe_unwrap(x) === unsafe_unwrap(y)
end

"""
    py_cast(t, py::Py)

Cast python object to julia.
"""
function py_cast(::Type{Nothing}, py::Py)
    py_coerce(Nothing, py)
end

function py_cast(::Type{Bool}, py::Py)
    py_equal_identity(py, PyAPI.Py_True) && return true
    py_equal_identity(py, PyAPI.Py_False) && return false
    return PyAPI.PyObject_IsTrue(py) != 0
end

function py_cast(::Type{T}, py::Py)::T where T <: Integer
    if PyAPI.PyNumber_Check(py) == 0
        py_seterror!(G_PyBuiltin.TypeError, "expected number type")
        py_throw()
    end
    intpy = Py(PyAPI.PyNumber_Long(py))
    if intpy == Py_NULLPTR
        py_throw()
    end
    py_coerce(T, intpy)
end

function py_cast(::Type{T}, py::Py)::T where T <: AbstractFloat
    d = PyAPI.PyFloat_AsDouble(py)
    if d == -1.0 && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    convert(T, d)
end

function py_cast(::Type{T}, py::Py)::T where T <: Complex
    d = PyAPI.PyComplex_AsCComplex(py) :: Py_complex
    if d.real == -1.0 && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    convert(T, complex(d.real, d.imag))
end

function py_cast(::Type{T}, py::Py)::T where T <: AbstractString
    py_coerce(T, py)
end

@generated function py_cast(::Type{T}, py::Py)::T where T <: Tuple
    T === Tuple && return :(error("$T should not equal to Tuple"))
    T isa DataType || return :(error("$T is not a data type"))
    len_tuple = length(T.parameters)
    result = Expr(:tuple, [:($py_cast($t, $unsafe_pytuple_get(py, $(i - 1)))) for (i, t) in enumerate(T.parameters)]...)
    quote
        if $CPython.PyAPI.PyObject_IsInstance(py,  $CPython.PyAPI.PyTuple_Type) == 0
            $py_seterror!(G_PyBuiltin.TypeError, "expected a tuple")
            $py_throw()
        end
        $check_tuple_length(py, $len_tuple)
        $result
    end
end

function py_cast(::Type{TArray}, py::Py)::TArray where {TArray <: AbstractArray}
    py_coerce(TArray, py)
end

"""
    py_cast(Py, o)

Cast julia variable to python object.
"""
function py_cast(::Type{Py}, o::Tuple)
    n = length(o)
    vec = Vector{Py}(undef, n)
    unroll_do!(Val(n), o) do i, o
        vec[i] = py_cast(Py, o[i])
    end
    py_tuple_create(vec...)
end

function py_cast(::Type{Py}, o::Bool)
    o ? PyAPI.Py_True : PyAPI.Py_False
end

function py_cast(::Type{Py}, o::AbstractFloat)
    return Py(PyAPI.PyFloat_FromDouble(convert(Cdouble, o)))
end

function py_cast(::Type{Py}, o::Complex)
    return Py(PyAPI.PyComplex_FromCComplex(convert(Py_complex, o)))
end

function py_cast(::Type{Py}, o::Integer)
    Py(PyAPI.PyLong_FromLongLong(convert(Clonglong, o)))
end

function py_cast(::Type{Py}, o::String)
    return Py(PyAPI.PyUnicode_FromStringAndSize(o, ncodeunits(o)))
end

function py_cast(::Type{Py}, o::Nothing)
    return PyAPI.Py_None
end

function py_cast(::Type{Py}, o::Py)
    return o
end

function py_cast(::Type{Py}, o::T) where {T <: AbstractArray{<:Number}}
    py_coerce(Py, o)
end
