baremodule C

import Serialization
import Base
import Base: +, -, *, /
using Base: sizeof, Cvoid, Cstring, Cint, Type, Val, reinterpret, convert

const PTR_SIZE = sizeof(Base.Ptr{Cvoid}) * 8
primitive type Ptr{T} PTR_SIZE end

function Ptr{T}(ptr::Ptr) where T
    reinterpret(Ptr{T}, ptr)
end

function Ptr{T}(ptr::Base.Ptr) where T
    reinterpret(Ptr{T}, ptr)
end

function Ptr{T}() where T
    reinterpret(Ptr{T}, C_NULL)
end

function Ptr(ptr::Base.Ptr{T}) where T
    reinterpret(Ptr{T}, ptr)
end

function Ptr{T}(address::Integer) where T
    reinterpret(Ptr{T}, address)
end

Base.@inline Base.@generated function GEP(ptr::Ptr{T}, ::Val{S}) where {T, S}
    offset = Base.fieldoffset(T, Base.fieldindex(T, S))
    fieldtype = Base.fieldtype(T, S)
    :($Ptr{$fieldtype}(ptr + $offset))
end

Base.convert(::Type{P}, ptr::Ptr) where P <: Base.Ptr = reinterpret(P, ptr)
Base.convert(::Type{P}, ptr::Base.Ptr) where P <: Ptr = reinterpret(P, ptr)
Base.convert(::Type{P}, ptr::UInt) where P <: Ptr = reinterpret(P, ptr)

Base.@inline Base.getproperty(ptr::Ptr, symbol::Symbol) = GEP(ptr, Val(symbol))
Base.@inline Base.getindex(ptr::Ptr{T}) where T = Base.unsafe_load(reinterpret(Base.Ptr{T}, ptr))
Base.@inline Base.setindex!(ptr::Ptr{T}, v) where T = Base.unsafe_store!(reinterpret(Base.Ptr{T}, ptr), convert(T, v))

function (ptr::Ptr{T} + i::Integer) where T
    Ptr{T}(reinterpret(Base.Ptr{Cvoid}, ptr) + i)
end

is_nullptr(x::C.Ptr) = reinterpret(Base.Ptr{Cvoid}, x) === Base.C_NULL

function Serialization.serialize(s::Serialization.AbstractSerializer, o::Ptr)
    Serialization.writetag(s, Serialization.OBJECT_TAG)
    Base.Serializer.serialize(s, Ptr)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{P}) where P <: Ptr
    return P(C_NULL)
end

end