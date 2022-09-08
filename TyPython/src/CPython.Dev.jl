#=
implementing necessary utilities to create CPython extensions.
=#
using TyPython.Reflection
import TyPython
import TyPython.Utils
import MacroTools: @q
export @export_py, @export_pymodule, Pyfunc

const refl = Reflection

const METH_CFUNC = METH_FASTCALL # METH_VARARGS | METH_KEYWORDS

"""
Return a new PyObject from a Julia function, if the latter has been marked using `@export_py`.
"""
function Pyfunc(f)
    error("there is no python C API function binding to $f")
end

macro export_py(ex)
    fi :: FuncInfo = parse_function(__source__, ex)
    esc(export_py(__module__, __source__, fi))
end

function _errormsg_argmismatch(args::Py, nargs::Integer)
    ngot = length(args)
    return "expected $nargs arguments, got $ngot."
end

function _errormsg_argmismatch(argc::Integer, nargs::Integer)
    return "expected $nargs arguments, got $argc."
end

_to_py_error(::Exception) = G_PyBuiltin.RuntimeError
_to_py_error(::MethodError) = G_PyBuiltin.TypeError
_to_py_error(::BoundsError) = G_PyBuiltin.IndexError
_to_py_error(::KeyError) = G_PyBuiltin.KeyError
_to_py_error(::ArgumentError) = G_PyBuiltin.ValueError
_to_py_error(::DimensionMismatch) = G_PyBuiltin.ValueError
_to_py_error(::UndefVarError) = G_PyBuiltin.NameError
_to_py_error(::EOFError) = G_PyBuiltin.EOFError
_to_py_error(::OutOfMemoryError) = G_PyBuiltin.MemoryError
_to_py_error(::OverflowError) = G_PyBuiltin.OverflowError

const PyPtr = Ptr{PyObject}

function export_py(__module__::Module, __source__::LineNumberNode, fi::FuncInfo)
    if !(fi.name isa Symbol)
        error("Python C API function name must be a valid symbol, not $(fi.name)!")
    end

    if fi.returnType isa refl.Undefined
        error("Python C API function must have a return type")
    end

    parTypes = Any[]
    for par in fi.pars
        if par.name isa refl.Undefined || par.type isa refl.Undefined
            error("Python C API function must have name and type for all parameters")
        end
        if par.defaultVal isa refl.Undefined
        else
            error("Python C API function cannot have default parameters")
        end
        push!(parTypes, par.type)
    end

    if !isempty(fi.kwPars)
        error("Python C API function does not have implemented keyword parameters")
    end

    if !isempty(fi.typePars)
        error("Python C API function does not have implemented generics")
    end

    nargs = length(fi.pars)

    pyfname = Symbol(string(fi.name), "##", "_pyfunc")
    pyfname_string_name = Symbol(string(fi.name), "##", "_name")
    pyfname_noinline = Symbol(string(fi.name), "##", "_pyfunc_noinline")
    pyfname_except = Symbol(string(fi.name), "##", "_pyfunc_except")
    pyfptrname = Symbol(string(fi.name), "##", "_pyfuncptr")
    pymethname = Symbol(string(fi.name), "##", "_pymethod_o")
    pyfuncobjectname = Symbol(string(fi.name), "##", "_pyo")
    pyfuncobjectname_ = Symbol(string(fi.name), "##", "_pyo_ptr")
    pydocname = Symbol(string(fi.name), "##", "_pydoc")

    orig = to_expr(fi)
    quote
        $Base.@__doc__ $orig

        Base.@noinline function $pyfname_noinline(self::$(Ptr{PyObject}), _vectorargs::$(Ptr{C.Ptr{PyObject}}), argc::$Py_ssize_t)::$(Ptr{PyObject})
            $__source__
            $__source__
            if argc != $nargs
                $CPython.PyAPI.PyErr_SetString($CPython.PyAPI.PyExc_ValueError[], $_errormsg_argmismatch(argc, $nargs))
                return $Py_NULLPTR
            end
            $([:(local $(Symbol("arg", i)) = $Py($BorrowReference(), $unsafe_load(_vectorargs, $i))) for i = 1:nargs]...)
            __o = $py_cast($Py, $(fi.name)($([:($py_coerce($(parTypes[i]), $(Symbol("arg", i)))) for i = 1:nargs]...)))
            $CPython.PyAPI.Py_IncRef(__o)
            return $unsafe_unwrap(__o)
        end

        Base.@noinline function $pyfname_except(e::Exception)
            if e isa $PyException
                $CPython.PyAPI.PyErr_SetObject(e.type, e.value)
            else
                msg = $Utils.capture_out() do
                    $Base.showerror(stderr, e, $catch_backtrace())
                end
                $CPython.PyAPI.PyErr_SetObject($_to_py_error(e), $py_cast($Py, msg))
            end
            return $Py_NULLPTR
        end

        function $pyfname(self::$(Ptr{PyObject}), _vectorargs::$(Ptr{C.Ptr{PyObject}}), argc::$Py_ssize_t)::$(Ptr{PyObject})
            $__source__
            $__source__
            try
                return $pyfname_noinline(self, _vectorargs, argc)
            catch e
                return $pyfname_except(e)
            end
        end


        const $pyfname_string_name = $(string(fi.name))
        const $pymethname = $Ref{$PyMethodDef}()


        const $pyfuncobjectname = $(Ref{Py})()
        const $pyfuncobjectname_ = $(Ref{Ptr{Cvoid}})($C_NULL)
        const $pydocname = Ref{String}()

        function $TyPython.CPython.Pyfunc(::typeof($(fi.name)))
            if $pyfuncobjectname_[] == $C_NULL
                $pyfptrname = $Base.@cfunction($pyfname, $PyPtr, ($PyPtr, $(Ptr{C.Ptr{PyObject}}), $Py_ssize_t))
                $pydocname[] = repr($Base.Docs.doc($(fi.name)))
                $pymethname[] = $PyMethodDef(
                    $pointer($pyfname_string_name),
                    $pyfptrname,
                    $METH_FASTCALL,
                    $Base.unsafe_convert($Cstring, $pydocname[])
                )
                $pyfuncobjectname[] = $Py($CPython.PyAPI.PyCFunction_NewEx($Base.unsafe_convert($(Ptr{Cvoid}), $pymethname), $C_NULL, $C_NULL))
                $pyfuncobjectname_[] = $reinterpret($(Ptr{Cvoid}), $unsafe_unwrap($pyfuncobjectname[]))
            end
            return $pyfuncobjectname[]
        end
        $(fi.name)
    end
end

macro export_pymodule(name::Symbol, ex)
    @switch ex begin
        @case Expr(:block, suite...)
        @case _
            error("@export_pymodule expects a begin-end block")
    end
    body = Expr(:block)
    out = Expr(:let, Expr(:block), body)
    sym_module = gensym("mod_$name")
    module_name = string(name)
    push!(body.args, :(local $sym_module = $CPython.G_PyBuiltin.__import__($py_cast($Py, "types")).SimpleNamespace()))
    for arg in suite
        @switch arg begin
            @case :($name = $value)
                push!(body.args, :($sym_module.$name = $py_cast($Py, $value)))
            @case ::LineNumberNode
                push!(body.args, arg)
            @case _
                error("@export_pymodule expects a block of `name = value` statements")
        end
    end
    push!(body.args, :($CPython.G_PyBuiltin.__import__($py_cast($Py, "sys")).modules[$py_cast($Py, $module_name)] = $sym_module))
    esc(out)
end
