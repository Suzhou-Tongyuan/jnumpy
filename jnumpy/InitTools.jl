module InitTools
import Pkg
import UUIDs

@nospecialize

function check_if_typython_installed(typython_dir::AbstractString)
    VERSION >= v"1.9" && error("Support for Julia 1.9 is coming soon.")
    VERSION < v"1.6" && error("TyPython works for Julia >= 1.6!")
    CTX = Pkg.API.Context()
    uuid_TyPython = UUIDs.UUID("9c4566a2-237d-4c69-9a5e-9d27b7d0881b")
    if !haskey(CTX.env.manifest.deps, uuid_TyPython)
        return false
    end
    pkgentry = CTX.env.manifest.deps[uuid_TyPython]::Pkg.Types.PackageEntry
    isnothing(pkgentry.path) && return false
    return abspath(typython_dir) == abspath(joinpath(dirname(CTX.env.project_file), pkgentry.path))
end

function _develop_typython(typython_dir::AbstractString)
    typython_dir = abspath(typython_dir)
    Pkg.develop(path=typython_dir)
    nothing
end

function setup_environment(typython_dir::AbstractString)
    if !check_if_typython_installed(typython_dir)
        _develop_typython(typython_dir)
        Pkg.resolve()
        Pkg.instantiate()
    end
    nothing
end


"""
The precompiled file goes wrong for unknown reason.
Removing and re-adding works.
"""
@noinline function force_resolve(typython_dir::AbstractString)
    try
        Pkg.rm("TyPython", io=devnull)
    catch
    end
    Pkg.develop(path=typython_dir, io=devnull)
    Pkg.resolve()
    Pkg.instantiate()
    nothing
end

@noinline function activate_project(project_dir::AbstractString, typython_dir::AbstractString)
    Pkg.activate(project_dir, io=devnull)
    force_resolve(typython_dir)
    nothing
end

end
