module Reflection
import TyPython: DevOnly
DevOnly.@devonly using MLStyle: @switch, @as_record

export TypeVarInfo, TypeParamInfo, ParamInfo, FuncInfo
export parse_typevar, parse_type_parameter, parse_parameter, parse_function, to_expr

if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@compiler_options"))
    @eval Base.Experimental.@compiler_options compile=min infer=no optimize=0
end

DevOnly.@staticinclude("Reflection.DevOnly.jl")

end
