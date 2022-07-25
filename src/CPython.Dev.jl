#=
implementing necessary utilities to create CPython extensions.
=#
using RawPython.Reflection
import RawPython.Utils
import MacroTools: @q
export @export_py, @export_pymodule, Pyfunc

const refl = Reflection

const METH_CFUNC = METH_VARARGS | METH_KEYWORDS

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
        push!(parTypes, par.type)
    end

    if !isempty(fi.kwPars)
        error("Python C API function does not have implemented keyword parameters")
    end

    if !isempty(fi.typePars)
        error("Python C API function does not have implemented generics")
    end

    nargs = length(fi.pars)
    argtypes = Expr(:curly, Tuple, parTypes...)

    pyfname = Symbol(string(fi.name), "##", "_pyfunc")
    pyfname_string_name = Symbol(string(fi.name), "##", "_name")
    pyfname_noinline = Symbol(string(fi.name), "##", "_pyfunc_noinline")
    pyfptrname = Symbol(string(fi.name), "##", "_pyfuncptr")
    pymethname = Symbol(string(fi.name), "##", "_pymethod_o")
    pyfuncobjectname = Symbol(string(fi.name), "##", "_pyo")
    pyfuncobjectname_ = Symbol(string(fi.name), "##", "_pyo_ptr")

    orig = to_expr(fi)
    quote
        $orig

        Base.@noinline function $pyfname_noinline(::$(Ptr{PyObject}), _py_args::$(Ptr{PyObject}), _py_kwargs::$(Ptr{PyObject}))::$(Ptr{PyObject})
            $__source__
            $__source__
            _py_args = $C.Ptr(_py_args)
            $PyAPI.Py_IncRef(_py_args)
            py_args = Py(_py_args)
            if length(py_args) != $nargs
                _err_msg = _errormsg_argmismatch(py_args, $nargs)
                println(_err_msg)
                flush(stdout)
                $PyAPI.PyErr_SetString($PyAPI.PyExc_ValueError[], _errormsg_argmismatch(py_args, $nargs))
                return $Py_NULLPTR
            end
            __o = py_cast($Py, $(fi.name)($py_coerce($argtypes, py_args)...))
            $PyAPI.Py_IncRef(__o)
            return $unsafe_unwrap(__o)
        end
        function $pyfname(self::$(Ptr{PyObject}), _py_args::$(Ptr{PyObject}), _py_kwargs::$(Ptr{PyObject}))::$(Ptr{PyObject})
            $__source__
            $__source__
            try
                return $pyfname_noinline(self, _py_args, _py_kwargs)
            catch e
                if e isa $PyException
                    $PyAPI.PyErr_SetObject(e.type, e.value)
                else
                    msg = $Utils.capure_out() do
                        $Base.showerror(stderr, e, catch_backtrace())
                    end
                    $PyAPI.PyErr_SetString($PyAPI.PyErr_SetObject, $CPython.G_PyBuiltin.Exception($py_cast($Py, msg)))
                end
                return $Py_NULLPTR
            end
        end
        const $pyfptrname = $Base.@cfunction($(Expr(:$, pyfname)), $PyPtr, ($PyPtr, $PyPtr, $PyPtr))
        const $pyfname_string_name = $(string(fi.name))
        const $pymethname = $Ref{$PyMethodDef}()


        const $pyfuncobjectname = Ref{Py}()
        const $pyfuncobjectname_ = Ref{Ptr{Cvoid}}(C_NULL)

        function RawPython.CPython.Pyfunc(::typeof($(fi.name)))
            if $pyfuncobjectname_[] == $C_NULL
                $pymethname[] = $PyMethodDef(
                    $pointer($pyfname_string_name),
                    $pyfptrname.ptr,
                    $METH_CFUNC,
                    $C_NULL
                )
                $pyfuncobjectname[] = $Py($PyAPI.PyCFunction_NewEx($Base.unsafe_convert($(Ptr{Cvoid}), $pymethname), $C_NULL, $C_NULL))
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
