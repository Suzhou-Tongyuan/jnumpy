module Reflection
using MLStyle
export TypeVarInfo, TypeParamInfo, ParamInfo, FuncInfo
export parse_typevar, parse_type_parameter, parse_parameter, parse_function, to_expr

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@compiler_options"))
    @eval Base.Experimental.@compiler_options compile=min infer=no optimize=0
end

function create_exception(ln::LineNumberNode, reason::String)
    LoadError(string(ln.file), ln.line, ErrorException(reason))
end

"""
    o isa Undefined

Representing undefined parameters/fields in Reflection.jl.
"""
struct Undefined end

const NullSymbol = Union{Symbol, Undefined}
const _undefined = Undefined()
const PVec{T, N} = NTuple{N, T}
const _pseudo_line = LineNumberNode(1)

"""
Converting `FuncInfo`/`TypeParamInfo`/`ParamInfo`/`TypeVarInfo` to `Expr`
"""
function to_expr end

"""
Representing a type variable in static type parameters.
    e.g, in `f(...) where {A <: T}`, `A` will be parsed into `TypeVarInfo` via `parse_typevar`.

```
Base.@kwdef struct TypeVarInfo
    base :: Any = _undefined
    typePars :: Union{Nothing, PVec{TypeVarInfo}} = nothing
end
```
"""
Base.@kwdef struct TypeVarInfo
    base :: Any = _undefined
    typePars :: Union{Nothing, PVec{TypeVarInfo}} = nothing
end

MLStyle.@as_record TypeVarInfo

function to_expr(t::TypeVarInfo)
    t.typePars === nothing && return t.base
    if isempty(t.typePars)
        t.base
    else
        :($(t.base){$(to_expr.(t.typePars)...)})
    end
end

"""
Representing a function parameter in Expr.

    e.g., `function f(param1::Int; param2::Int)`

```
Base.@kwdef mutable struct ParamInfo
    name :: Any = _undefined
    type :: Any = _undefined
    defaultVal :: Any = _undefined
    meta :: Vector{Any} = []
    isVariadic :: Bool = false
end
```
"""
Base.@kwdef mutable struct ParamInfo
    name :: Any = _undefined
    type :: Any = _undefined
    defaultVal :: Any = _undefined
    meta :: Vector{Any} = []
    isVariadic :: Bool = false
end

MLStyle.@as_record ParamInfo

function to_expr(p::ParamInfo)
    res = if p.name isa Undefined
        @assert !(p.type isa Undefined)
        :(::$(p.type))
    else
        if p.type isa Undefined
            p.name
        else
            :($(p.name)::$(p.type))
        end
    end
    if p.isVariadic
        res = Expr(:..., res)
    end
    if !(p.defaultVal isa Undefined)
        res = Expr(:kw, res, p.defaultVal)
    end
    if !isempty(p.meta)
        res = Expr(:meta, p.meta..., res)
    end
    return res
end


"""
Representing a static type parameter in Expr.

    e.g., `function f(::STP) where STP <: Integer`

```
Base.@kwdef struct TypeParamInfo
    name :: Symbol
    lb :: Union{TypeVarInfo, Undefined} = _undefined
    ub :: Union{TypeVarInfo, Undefined} = _undefined
end
```
"""
Base.@kwdef struct TypeParamInfo
    name :: Symbol
    lb :: Union{TypeVarInfo, Undefined} = _undefined
    ub :: Union{TypeVarInfo, Undefined} = _undefined
end

MLStyle.@as_record TypeParamInfo

function to_expr(tp::TypeParamInfo)
    if tp.lb isa Undefined
        if tp.ub isa Undefined
            tp.name
        else
            :($(tp.name) <: $(to_expr(tp.ub)))
        end
    else
        if tp.ub isa Undefined
            :($(tp.name) >: $(to_expr(tp.lb)))
        else
            :($(to_expr(tp.lb)) <: $(tp.name) <: $(to_expr(tp.ub)))
        end
    end
end


"""
Representing a function in Expr.
    e.g,
    - `function f(args...)`
    - `x -> x`
    - `(x, y) -> x`
    - `f(x) = x + 1`

```
Base.@kwdef mutable struct FuncInfo
    ln :: LineNumberNode = _pseudo_line
    name :: Any = _undefined
    pars :: Vector{ParamInfo} = ParamInfo[]
    kwPars :: Vector{ParamInfo} = ParamInfo[]
    typePars :: Vector{TypeParamInfo} = TypeParamInfo[]
    returnType :: Any = _undefined # can be _undefined
    body :: Any = _undefined # can be _undefined
    isAbstract :: Bool = false
end
```
"""
Base.@kwdef mutable struct FuncInfo
    ln :: LineNumberNode = _pseudo_line
    name :: Any = _undefined
    pars :: Vector{ParamInfo} = ParamInfo[]
    kwPars :: Vector{ParamInfo} = ParamInfo[]
    typePars :: Vector{TypeParamInfo} = TypeParamInfo[]
    returnType :: Any = _undefined # can be _undefined
    body :: Any = _undefined # can be _undefined
    isAbstract :: Bool = false
end

MLStyle.@as_record FuncInfo

function to_expr(f::FuncInfo)
    if f.isAbstract
        return :nothing
    else
        args = []
        if !isempty(f.kwPars)
            kwargs = Expr(:parameters)
            push!(args, kwargs)
            for each in f.kwPars
                push!(kwargs.args, to_expr(each))
            end
        end
        for each in f.pars
            push!(args, to_expr(each))
        end
        header = if f.name isa Undefined
           Expr(:tuple, args...)
        else
            Expr(:call, f.name, args...)
        end
        if !(f.returnType isa Undefined)
            header = :($header :: $(f.returnType))
        end
        if !isempty(f.typePars)
            header = :($header where {$(to_expr.(f.typePars)...)})
        end
        return Expr(:function, header, f.body)
    end
end

"""
    parse_typevar(ln::LineNumberNode, repr)

Parse a julia expression into `TypeVarInfo`, and an exception is thrown on failure.
"""
function parse_typevar(ln::LineNumberNode, repr)
    @switch repr begin
        @case :($typename{$(generic_params...)})
            return TypeVarInfo(typename, Tuple(parse_typevar(ln, x) for x in generic_params))
        @case typename
            return TypeVarInfo(typename, nothing)
        @case _
            throw(create_exception(ln, "invalid type representation: $repr"))
    end
end


"""
    parse_parameter(ln::LineNumberNode, p; support_tuple_parameters=true)

Parse a julia expression into `ParamInfo`, and an exception is thrown on failure.
"""
function parse_parameter(ln::LineNumberNode, p; support_tuple_parameters=true)
    self = ParamInfo()
    parse_parameter!(ln, self, p, support_tuple_parameters)
    return self
end

function parse_parameter!(ln :: LineNumberNode, self::ParamInfo, p, support_tuple_parameters)
    @switch p begin
        @case Expr(:meta, x, p)
            push!(self.meta, x)
            parse_parameter!(ln, self, p, support_tuple_parameters)
        @case Expr(:..., p)
            self.isVariadic = true
            parse_parameter!(ln, self, p, support_tuple_parameters)
        @case Expr(:kw, p, b)
            self.defaultVal = b
            parse_parameter!(ln, self, p, support_tuple_parameters)
        @case :(:: $t)
            self.type = t
            nothing
        @case :($p :: $t)
            self.type = t
            parse_parameter!(ln, self, p, support_tuple_parameters)
        @case p::Symbol
            self.name = p
            nothing
        @case Expr(:tuple, _...)
            if support_tuple_parameters
                self.name = p
            else
                throw(create_exception(ln, "tuple parameters are not supported"))
            end
            nothing
        @case _
            throw(create_exception(ln, "invalid parameter $p"))
    end
end

"""
    parse_type_parameter(ln::LineNumberNode, t)

Parse a julia expression into `TypeParamInfo`, and an exception is thrown on failure.
"""
function parse_type_parameter(ln::LineNumberNode, t)
    @switch t begin
        @case :($lb <: $(t::Symbol) <: $ub) || :($ub >: $(t::Symbol) >: $lb)
            TypeParamInfo(t, parse_typevar(ln, lb), parse_typevar(ln, ub))
        @case :($(t::Symbol) >: $lb)
            TypeParamInfo(t, parse_typevar(ln, lb), _undefined)
        @case :($(t::Symbol) <: $ub)
            TypeParamInfo(t, _undefined, parse_typevar(ln, ub))
        @case t::Symbol
            TypeParamInfo(t, _undefined, _undefined)
        @case _
            throw(create_exception(ln, "invalid type parameter $t"))
    end
end


"""
    parse_function(ln :: LineNumberNode, ex; fallback::T=_undefined,  allow_short_func::Bool=false, allow_lambda::Bool=false) where T

Parse a julia expression into `FuncInfo`, the parameter `fallback` is returned on failure.

If `fallback` is not given, an exception is thrown.
"""
function parse_function(ln :: LineNumberNode, ex; fallback::T=_undefined,  allow_short_func::Bool=false, allow_lambda::Bool=false) where T
    self :: FuncInfo = FuncInfo()
    @switch ex begin
        @case Expr(:function, header, body)
            self.body = body
            self.isAbstract = false # unnecessary but clarified
            parse_function_header!(ln, self, header; is_lambda = false, allow_lambda = allow_lambda)
            return self
        @case Expr(:function, header)
            self.isAbstract = true
            parse_function_header!(ln, self, header; is_lambda = false, allow_lambda = allow_lambda)
            return self
        @case Expr(:(->), header, body)
            if !allow_lambda
                throw(create_exception(ln, "lambda functions are not allowed here: $ex"))
            end
            self.body = body
            self.isAbstract = false
            parse_function_header!(ln, self, header; is_lambda = true, allow_lambda = true)
            return self
        @case Expr(:(=), Expr(:call, _...) && header, rhs)
            if !allow_short_func
                throw(create_exception(ln, "short functions are not allowed here: $ex"))
            end
            self.body = rhs
            self.isAbstract = false
            parse_function_header!(ln, self, header; is_lambda = false, allow_lambda = false)
            return self
        @case _
            if fallback isa Undefined
                throw(create_exception(ln, "invalid function expression: $ex"))
            else
                fallback
            end
    end
end

function parse_function_header!(ln::LineNumberNode, self::FuncInfo, header; is_lambda :: Bool = false, allow_lambda :: Bool = false)
    typePars = self.typePars

    @switch header begin
        @case Expr(:where, header, tyPar_exprs...)
            for tyPar_expr in tyPar_exprs
                push!(typePars, parse_type_parameter(ln, tyPar_expr))
            end
        @case _
    end

    @switch header begin
        @case Expr(:(::), header, returnType)
            self.returnType = returnType
        @case _
    end

    if is_lambda && !Meta.isexpr(header, :tuple)
        header = Expr(:tuple, header)
    end

    @switch header begin
        @case Expr(:call, f, Expr(:parameters, kwargs...), args...)
            for x in kwargs
                push!(self.kwPars, parse_parameter(ln, x))
            end
            for x in args
                push!(self.pars, parse_parameter(ln, x))
            end
            self.name = f
        @case Expr(:call, f, args...)
            for x in args
                push!(self.pars, parse_parameter(ln, x))
            end
            self.name = f
        @case Expr(:tuple, Expr(:parameters, kwargs...), args...)
            if !allow_lambda
                throw(create_exception(ln, "tuple function signature are not allowed here."))
            end
            for x in kwargs
                push!(self.kwPars, parse_parameter(ln, x))
            end
            for x in args
                push!(self.pars, parse_parameter(ln, x))
            end
        @case Expr(:tuple, args...)
            if !allow_lambda
                throw(create_exception(ln, "tuple function signature are not allowed here."))
            end
            for x in args
                push!(self.pars, parse_parameter(ln, x))
            end
        @case _
            if !self.isAbstract
                throw(create_exception(ln, "unrecognised function signature $header."))
            else
                self.name = header
            end
    end
end

end
