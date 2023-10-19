"""
    from_ndarray(x::Py)

Convert numpy ndarray to julia array,
ndarray with F-contiguous could be converted to Array without copy,
ndarray with C-contiguous could be converted to Transpose of Array(2D) or PermutedDimsArray of Array(high dimensions) without copy,
other ndarray will first be copied to f-contiguous ndarray, then converted to Array.
"""
function from_ndarray(x::Py)
    np = get_numpy()
    if PyAPI.PyObject_HasAttr(x, attribute_symbol_to_pyobject(:__array_struct__)) != 1
        error("from_ndarray: $x has no __array_struct__")
    end
    __array_struct__ = x.__array_struct__
    ptr = C.Ptr{PyArrayInterface}(_get_capsule_ptr(__array_struct__))
    info = ptr[] :: PyArrayInterface
    info.typekind in supported_kinds || error("unsupported numpy dtype: $(Char(ptr.typekind))")
    flags = info.flags
    if (!(checkbit(flags, NPY_ARRAY_F_CONTIGUOUS) || checkbit(flags, NPY_ARRAY_C_CONTIGUOUS)) ||
        !checkbit(flags, TYPY_SUPPORTED_NO_COPY_NP_FLAG))
        x = np.copy(x, order=py_cast(Py, "F"))
        __array_struct__ = x.__array_struct__
        ptr = C.Ptr{PyArrayInterface}(_get_capsule_ptr(__array_struct__))
        info = ptr[] :: PyArrayInterface
        flags = info.flags
    end

    # TODO: exception check
    shape = Tuple(Int(i)
        for i in unsafe_wrap(Array, convert(Ptr{Py_intptr_t}, info.shape), Int(info.nd); own=false)) :: ShapeType

    if checkbit(flags, NPY_ARRAY_C_CONTIGUOUS) && !(checkbit(flags, NPY_ARRAY_F_CONTIGUOUS))
        shape = reverse(shape)
    end

    arr = @switch (Char(info.typekind), Int(info.itemsize)) begin
        @case ('b', 1)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{Bool}, info.data), shape; own=false))
        @case ('i', 1)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{Int8}, info.data), shape; own=false))
        @case ('i', 2)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{Int16}, info.data), shape; own=false))
        @case ('i', 4)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{Int32}, info.data), shape; own=false))
        @case ('i', 8)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{Int64}, info.data), shape; own=false))
        @case ('f', 2)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{Float16}, info.data), shape; own=false))
        @case ('f', 4)
                register_root(x,
                    unsafe_wrap(Array, convert(Ptr{Float32}, info.data), shape; own=false))
        @case ('f', 8)
                register_root(x,
                    unsafe_wrap(Array, convert(Ptr{Float64}, info.data), shape; own=false))
        @case ('u', 1)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{UInt8}, info.data), shape; own=false))
        @case ('u', 2)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{UInt16}, info.data), shape; own=false))
        @case ('u', 4)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{UInt32}, info.data), shape; own=false))
        @case ('u', 8)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{UInt64}, info.data), shape; own=false))
        @case ('c', 4)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{ComplexF16}, info.data), shape; own=false))
        @case ('c', 8)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{ComplexF32}, info.data), shape; own=false))
        @case ('c', 16)
            register_root(x,
                unsafe_wrap(Array, convert(Ptr{ComplexF64}, info.data), shape; own=false))
        @case (code, nbytes)
            throw(error("unsupported numpy dtype: $(code) $(nbytes)"))
    end

    if checkbit(flags, NPY_ARRAY_C_CONTIGUOUS) && !(checkbit(flags, NPY_ARRAY_F_CONTIGUOUS))
        if info.nd == 2
            arr = transpose(arr)
        elseif info.nd > 2
            perm = Tuple(reverse(1:info.nd))
            arr = PermutedDimsArray(arr, perm)
        end
    end
    return arr
end
