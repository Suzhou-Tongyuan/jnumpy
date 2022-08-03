module Utils
import IOCapture
import TOML
import Pkg
function capture_out(f)
    capture = IOCapture.capture() do
        f()
    end
    capture.output
end

@generated function unroll_do!(f, ::Val{N}, args...) where N
    block = Expr(:block)
    for i = 1:N
        push!(block.args, :(f($i, args...)))
    end
    block
end

function add_deps(toml_path::String)
    proj_toml = TOML.parsefile(toml_path)
    deps = get(proj_toml, "deps", Dict())
    current_deps = Pkg.project().dependencies
    current_deps = Dict(k=>string(v) for (k,v) in pairs(current_deps))
    new_deps = merge(current_deps, deps)
    current_deps_path = Pkg.project().path
    open(current_deps_path, "w") do io
        data = TOML.parsefile(current_deps_path)
        data["deps"] = new_deps
        TOML.print(io, data)
    end
    Pkg.resolve()
    Pkg.instantiate()
end

end