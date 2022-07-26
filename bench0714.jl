module bench0714
using MKL

NumOrVec{Num} = Union{Num, AbstractArray{<:Num, 1}}

struct IsScalar end
struct IsArray end
struct NotScalar end

trait_scalarity(_) = NotScalar()
trait_scalarity(::Type{<:Number}) = IsScalar()
trait_scalarity(::Type{<:AbstractArray}) = IsArray()

"""convert appropriately data to builtin array
"""
asarray(x::T) where T = asarray(x, trait_scalarity(T))
asarray(x, ::IsScalar) = [x]
asarray(x, ::IsArray) = x
asarray(x, ::NotScalar) = error("$(typeof(x)) is not a scalar or array")

function _atleast_nd(x :: AbstractArray{T, N_src}, ::Val{N_dest}) where {T, N_src, N_dest}
    if N_src < N_dest
        reshape(x, (ntuple(_ -> 1, N_dest - N_src)..., size(x)...))
    else
        x
    end
end

function _atleast_nd(x, target_dim::Val)
    _atleast_nd(asarray(x), target_dim)
end

function atleast_1d(x)
    _atleast_nd(x, Val(1))
end

function atleast_2d(x)
    _atleast_nd(x, Val(2))
end

function atleast_3d(x)
    _atleast_nd(x, Val(3))
end


ravel(x::AbstractVector) = x
ravel(x::AbstractArray) = Iterators.flatten(x) |> collect

# for users
@inline function append(arr, values; ndim::Union{Nothing, Int} = nothing)
    append(asarray(arr), asarray(values), ndim=ndim)
end

# core logic
@inline function append(arr::AbstractArray, values::AbstractArray; ndim::Union{Nothing, Int} = nothing)
    if ndim === nothing
        if ndims(arr) !== 1
            arr = ravel(arr)
        end
        values = ravel(values)
        ndim = ndims(arr)
    end

    # a bit magic for inferring types for 'cat'
    maxdim = max(ndims(arr), ndims(values))
    array_elty = Base.promote_type(eltype(arr), eltype(values))
    result = cat(arr, values, dims=ndim)
    if ndim > maxdim
        result :: Array{array_elty, ndim}
    else
        result :: Array{array_elty, maxdim}
    end
end

castarray(::Type{E}, x::AbstractArray{<:E, 1}) where E = x

function castarray(::Type{E}, x::AbstractArray{<:E, N}) where {E, N}
    shape = size(x)
    if count(x -> x > 1, shape) > 1
        error("more than 1 dimensions have data")
    end
    i = findfirst(x -> x != 1, shape)
    if i === nothing
        return @view x[1:1]
    end
    return view(x, map(x -> x > 1 ? (1:x) : 1, shape)...)
end

castarray(::Type{E}, x::AbstractArray) where E = collect(E, x)

"""
Return a digital IIR filter from an analog one using a bilinear transform.

Transform a set of poles and zeros from the analog s-plane to the digital
z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
``s``, maintaining the shape of the frequency response.
"""
function bilinear_zpk end

_realtype(::Type{Complex{F}}) where F = F
_realtype(::Type{T}) where T<:Number = T

# for users
function bilinear_zpk(z::AbstractVecOrMat{A}, p::AbstractVecOrMat{B}, k::C, fs::D) where {
        A <: Number,
        B <: Number,
        C <: Number,
        D <: Number
    }

    E = Base.promote_type(_realtype(A), _realtype(B), C, D, Float32)
    z = castarray(Complex{E}, z)
    p = castarray(Complex{E}, p)
    k = convert(E, k) 
    fs′ = convert(E, fs)
    bilinear_zpk(z, p, k, fs′)
end

function bilinear_zpk(z::AbstractVector{Complex{F}}, p::AbstractVector{Complex{F}}, k::F, fs::F) where {F <: AbstractFloat}
    z = atleast_1d(z)
    p = atleast_1d(p)
    degree = _relative_degree(z, p)
    fs2 = convert(F, 2 * fs)
    z_z = @fastmath (fs2 .+ z) ./ (fs2 .- z)
    p_z = @fastmath (fs2 .+ p) ./ (fs2 .- p)
    z_z = append(z_z, -ones(F, degree))
    k_z = k * real(prod(@fastmath fs2 .- z) / prod(@fastmath fs2 .- p))
    return z_z, p_z, k_z
end

# core logic
"""
Return relative degree of transfer function from zeros and poles
"""
function _relative_degree(z, p)    
    degree = length(p) - length(z)
    if degree < 0
        error("Improper transfer function. " *
              "Must have at least as many poles as zeros.")
    else
        return degree
    end
end

end # module