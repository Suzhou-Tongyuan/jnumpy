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
