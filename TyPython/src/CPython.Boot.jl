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
    G_IsInitialized[] && return
    if haskey(ENV, CF_TYPY_PY_APIPTR)
        ptr = reinterpret(Ptr{Cvoid}, parse(UInt, ENV[CF_TYPY_PY_APIPTR]))
        init(ptr)
    elseif haskey(ENV, CF_TYPY_PY_DLL)
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
    G_IsInitialized[] = false
    return
end

const _init_indicator = Ref(C_NULL)

function init(ptr :: Ptr{Cvoid})
    _init_indicator[] != C_NULL && return

    init_api!(ptr)
    G_IsInitialized[] = true
    atexit() do
        if G_IsInitialized[]
            WITH_GIL() do
                G_IsInitialized[] = false
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

        _init_indicator[] = Ptr{Cvoid}(12312) # any non-zero number is fine
    end
end
