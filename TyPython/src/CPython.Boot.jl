import Libdl

function init_api!(@nospecialize(pythonapi :: Ptr{Cvoid}))
    for (name, ftype) in zip(fieldnames(PythonAPIStruct), fieldtypes(PythonAPIStruct))
        ftype === Py && continue
        dlsym = Symbol(replace(string(name), r"(.*)(_variant.*)" => s"\1"))
        setfield!(PyAPI, name, convert(ftype, Libdl.dlsym(pythonapi, dlsym)))
    end
end

function init_values!(py_builtin_module::Py)
    for (name, ftype) in zip(fieldnames(PythonAPIStruct), fieldtypes(PythonAPIStruct))
        ftype !== Py && continue
        module_field = Symbol(replace(string(name), r"Py_(.*)" => s"\1"))
        setfield!(PyAPI, name, getproperty(py_builtin_module, module_field))
    end
end

function load_pydll!(dllpath::AbstractString)
    cd(dirname(dllpath))
    return Libdl.dlopen(convert(String, dllpath), Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL)
end

function init()
    RT_is_initialized() && return
    if haskey(ENV, CF_TYPY_PY_APIPTR)
        if check_pid()
            ptr = reinterpret(Ptr{Cvoid}, parse(UInt, ENV[CF_TYPY_PY_APIPTR]))
            init(ptr)
        else
            pop!(ENV, CF_TYPY_PY_APIPTR)
            pop!(ENV, CF_TYPY_PID)
            init()
        end
        return
    end
    if haskey(ENV, CF_TYPY_PY_DLL)
        cwd = pwd()
        try
            ptr = load_pydll!(ENV[CF_TYPY_PY_DLL])
            init(ptr)
        finally
            cd(cwd)
        end
    else
        error("Python shared library not found: try setting the environment variable $(CF_TYPY_PY_APIPTR) or $(CF_TYPY_PY_DLL).")
    end
end

function _atpyexit()
    RT_set_dead!()
    return
end


function init(ptr :: Ptr{Cvoid})
    RT_is_initialized() && return

    init_api!(ptr)
    RT_set_configuration!()
    atexit() do
        RT_set_dead!()
        if RT_is_initialized() && PyAPI.Py_IsInitialized() != 0
            WITH_GIL() do
                PyAPI.Py_FinalizeEx()
            end
        end
    end
    WITH_GIL() do
        if is_calling_julia_from_python()
            if PyAPI.Py_IsInitialized() == 0
                error("Python is not initialized from python?")
            end
        else
            PyAPI.Py_InitializeEx(0)
        end
        builtins = PyAPI.PyImport_ImportModule("builtins")
        unsafe_set!(G_PyBuiltin, builtins)
        init_values!(G_PyBuiltin)
        if PyAPI.Py_AtExit(@cfunction(_atpyexit, Cvoid, ())) == -1
            @warn "Py_AtExit() error"
        end
        if !is_calling_julia_from_python()
            sys = py_import("sys")
            sys.argv.append(py_cast(Py, "python"))
            nothing
        end
    end
end
