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
    if haskey(ENV, CF_RAWPY_PY_APIPTR)
        ptr = reinterpret(Ptr{Cvoid}, parse(UInt, ENV[CF_RAWPY_PY_APIPTR]))
        init(ptr)
    elseif haskey(ENV, CF_RAWPY_PY_DLL)
        cwd = pwd()
        try
            ptr = load_pydll!(ENV[CF_RAWPY_PY_DLL])
            init(ptr)
        finally
            cd(cwd)
        end
    else
        error("Python shared library not found: try setting the environment variable $(CF_RAWPY_PY_APIPTR) or $(CF_RAWPY_PY_DLL).")
    end
end

function init(ptr :: Ptr{Cvoid})
    init_api!(ptr)
    G_IsInitialized[] = true
    atexit() do
        WITH_GIL() do
            PyAPI.Py_Finalize()
            G_IsInitialized[] = false
        end
    end
    WITH_GIL() do
        if is_calling_julia_from_python()
        else
            PyAPI.Py_InitializeEx(0)
        end
        builtins = PyAPI.PyImport_ImportModule("builtins")
        unsafe_set!(G_PyBuiltin, builtins)
        init_values!(G_PyBuiltin)
    end
end
