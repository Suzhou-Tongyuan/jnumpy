struct DynamicArray
    arr :: Any
    eltype :: DataType
    shape :: Vector{Py_ssize_t}
    strides :: Vector{Py_ssize_t}
    ptr :: Ptr{Cvoid}
    ndim :: Cint
    itemsize :: Cint
    typekind :: Cchar
    is_c_style :: Bool
    is_f_style :: Bool
    need_permute :: Bool
    perm :: NTuple{N, Int} where N
end

const G_arrayinfo = Union{DynamicArray, Nothing}[]
const G_arrayinfo_unused_slots = Int[]
const G_pymodule_types = Ref{Py}()

function py_getmodule_types()
    if py_isnull(G_pymodule_types[])
        G_pymodule_types[] = G_PyBuiltin.__import__(py_cast(Py, "types"))
    end
    return G_pymodule_types[]
end

const PRE_CACHE_SIZE = 500
function __init_julia_wrap__()
    empty!(G_arrayinfo)
    empty!(G_arrayinfo_unused_slots)
    for i in 1:PRE_CACHE_SIZE
        push!(G_arrayinfo, nothing)
        push!(G_arrayinfo_unused_slots, i)
    end
    G_pymodule_types[] = Py(Py_NULLPTR)
end

function LinearAlgebra.transpose(d::DynamicArray)
    if d.is_c_style || d.is_f_style
        DynamicArray(d.arr, d.eltype, reverse(d.shape), reverse(d.strides), d.ptr, d.ndim, d.itemsize, d.typekind, !d.is_c_style, !d.is_f_style, false, d.perm)
    else
        DynamicArray(d.arr, d.eltype, reverse(d.shape), reverse(d.strides), d.ptr, d.ndim, d.itemsize, d.typekind, false, false, false, d.perm)
    end
end

function permutedims(d::DynamicArray, perm)
    DynamicArray(d.arr, d.eltype, d.shape, d.strides, d.ptr, d.ndim, d.itemsize, d.typekind, d.is_c_style, d.is_f_style, true, perm)
end

function get_perm(x::PermutedDimsArray{T, N, perm}) where {T, N, perm}
    return perm .- 1
end

function get_typekind(_::Bool)
    return Cchar('b')
end

function get_typekind(_::Union{Int8, Int16, Int32, Int64})
    return Cchar('i')
end

function get_typekind(_::Union{UInt8, UInt16, UInt32, UInt64})
    return Cchar('u')
end

function get_typekind(_::Union{Float16, Float32, Float64})
    return Cchar('f')
end

function get_typekind(_::Union{ComplexF16, ComplexF32, ComplexF64})
    return Cchar('c')
end

function _normalized_array(x::AbstractArray, is_f_style)
    normalized_x = if x isa Array
        x
    elseif x isa StridedArray && x isa SubArray
        is_f_style = Base.iscontiguous(x)
        x
    elseif x isa StridedArray && x isa Base.ReinterpretArray && parent(x) isa Array
        x
    else
        collect(x)
    end
    return normalized_x, is_f_style
end

function DynamicArray(x::AbstractArray)
    et = eltype(x)
    typekind = get_typekind(zero(et))
    if x isa LinearAlgebra.Transpose && parent(x) isa StridedArray
        return LinearAlgebra.transpose(DynamicArray(LinearAlgebra.transpose(x)))
    elseif x isa PermutedDimsArray && parent(x) isa StridedArray
        return permutedims(DynamicArray(parent(x)), get_perm(x))
    end
    is_c_style = false
    is_f_style = true
    normalized_x, is_f_style = _normalized_array(x, is_f_style)
    shape = collect(Py_ssize_t, size(normalized_x))
    ptr = reinterpret(Ptr{Cvoid}, pointer(normalized_x))
    ndim = convert(Cint, ndims(normalized_x))
    itemsize = convert(Cint, sizeof(et))
    np_strides = Py_ssize_t[ itemsize * d for d in strides(normalized_x) ]
    perm = Tuple(0:ndim-1)
    DynamicArray(normalized_x, et, shape, np_strides, ptr, ndim, itemsize, typekind, is_c_style, is_f_style, false, perm)
end

# handle GC
struct ArrayInterfaceCapsule
    array_interface :: PyArrayInterface
    array_info_slot :: Int
end

function PyCapsule_Destruct_DynamicArray(o::Ptr{PyObject})::Cvoid
    o = PyAPI.PyCapsule_GetPointer(C.Ptr(o), reinterpret(Cstring, C_NULL))
    if o == C_NULL && PyAPI.PyErr_Occurred() != Py_NULLPTR
        py_throw()
    end
    ptr = C.Ptr{ArrayInterfaceCapsule}(o)
    array_info_slot = ptr.array_info_slot[]
    push!(G_arrayinfo_unused_slots, array_info_slot)
    G_arrayinfo[array_info_slot] = nothing
    Base.Libc.free(o)
    nothing
end

function mk_capsule(val::DynamicArray)
    ptr_array_interface = reinterpret(C.Ptr{ArrayInterfaceCapsule}, Base.Libc.malloc(sizeof(ArrayInterfaceCapsule)))
    array_info_slot = if isempty(G_arrayinfo_unused_slots)
        push!(G_arrayinfo, nothing)
        length(G_arrayinfo)
    else
        pop!(G_arrayinfo_unused_slots)
    end
    ptr_array_interface[] =
        ArrayInterfaceCapsule(
            PyArrayInterface(
                nd = val.ndim,
                typekind = val.typekind,
                itemsize = val.itemsize,
                flags = (val.is_c_style ? NPY_ARRAY_C_CONTIGUOUS : Cint(0x0000)) | (val.is_f_style ? NPY_ARRAY_F_CONTIGUOUS : Cint(0x0000)) | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED,
                shape = pointer(val.shape),
                strides = pointer(val.strides),
                data = val.ptr
            ),
        array_info_slot
    )
    G_arrayinfo[array_info_slot] = val
    return Py(PyAPI.PyCapsule_New(
        reinterpret(Ptr{Cvoid}, ptr_array_interface),
        reinterpret(Cstring, C_NULL),
        @cfunction(PyCapsule_Destruct_DynamicArray, Cvoid, (Ptr{PyObject}, ))))
end

function py_coerce(::Type{Py}, @nospecialize(xs::DynamicArray))
    capsule = mk_capsule(xs)
    types = py_getmodule_types()
    np = get_numpy()
    nparray = np.asarray(types.SimpleNamespace(__array_struct__ = capsule))
    if xs.need_permute
        nparray = nparray.transpose(py_cast(Py, xs.perm))
    end
    return nparray
end

"""
    py_coerce(::Type{Py}, xs::AbstractArray)

Convert julia array to numpy ndarray.
Array, Transpose of Array, PermutedDimsArray of Array and some SubArray/ReinterpretArray could be converted to ndarray without copy,
other arrays will first be copied with `collect()`, then converted to ndarray.
"""
function py_coerce(::Type{Py}, @nospecialize(xs::AbstractArray))
    xs = DynamicArray(xs)
    py_coerce(Py, xs)
end
