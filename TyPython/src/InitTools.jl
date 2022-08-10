module InitTools

import Pkg
import UUIDs

function check_if_typython_installed(typython_path::AbstractString)
    VERSION >= v"1.9" && error("Support for Julia 1.9 is coming soon.")
    VERSION < v"1.7" && error("TyPython works for Julia >= 1.7!")
    CTX = Pkg.API.Context()
    uuid_TyPython = UUIDs.UUID("9c4566a2-237d-4c69-9a5e-9d27b7d0881b")
    if !haskey(CTX.env.manifest.deps, uuid_TyPython)
        return false
    end
    pkgentry = CTX.env.manifest.deps[uuid_TyPython]::Pkg.Types.PackageEntry
    isnothing(pkgentry.path) && return false
    return abspath(typython_path) == abspath(joinpath(dirname(CTX.env.project_file), pkgentry.path))
end

function _develop_typython(typython_path::AbstractString)
    typython_path = abspath(typython_path)
    Pkg.develop(path=typython_path)
    nothing
end

function setup_environment(typython_path::AbstractString)
    if !check_if_typython_installed(typython_path)
        _develop_typython(typython_path)
        Pkg.resolve()
        Pkg.instantiate()
    end
    nothing
end

function activate_project(project_dir::AbstractString, typython_path::AbstractString)
    Pkg.activate(project_path)
    setup_environment(typython_path)
    nothing
end

function print_project_name(project_dir::AbstractString)
    Pkg.activate(project_path, io=devnull)
    n = Pkg.project().name
    if !isnothing(n)
        println(n)
    end
    nothing
end

end
