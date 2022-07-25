CF_RAWPY_MODE = "RAWPY_MODE"
CF_RAWPY_PY_APIPTR = "RAWPY_PY_APIPTR"
CF_RAWPY_PY_DLL = "RAWPY_PY_DLL"
CF_RAWPY_JL_EXE = "RAWPY_JL_EXE"
CF_RAWPY_MODE_PYTHON = "PYTHON-BASED"
CF_RAWPY_MODE_JULIA = "JULIA-BASED"
CF_RAWPY_JL_OPTS = "RAWPY_JL_OPTS"

import os
import subprocess
import ctypes
import json
import shlex

julia_info_query = r"""
import Libdl
import Pkg
println(Base.Sys.BINDIR)
println(abspath(Libdl.dlpath("libjulia")))
println(unsafe_string(Base.JLOptions().image_file))
println(dirname(Pkg.project().path))
"""

def args_from_config(exepath: str, args: list[str]):
    args = [exepath] + args
    
    # python 2 is deprecated, we just consider python 3
    argv: list[bytes] = [arg.encode('utf-8') for arg in args] 
    argc = len(argv)
    argc = ctypes.c_int(argc)
    argv = ctypes.POINTER(ctypes.c_char_p)((ctypes.c_char_p * len(argv))(*argv))  # type: ignore
    return argc, argv



def eval_jl(x):
    global _eval_jl
    _eval_jl(x)  # type: ignore

def init_jl():
    global _eval_jl
    if os.getenv(CF_RAWPY_MODE) == CF_RAWPY_MODE_JULIA:
        return
    elif os.getenv(CF_RAWPY_MODE) == CF_RAWPY_MODE_PYTHON:
        assert os.getenv(CF_RAWPY_PY_APIPTR, str(ctypes.pythonapi._handle))
        return
    elif not os.getenv(CF_RAWPY_MODE):
        os.environ[CF_RAWPY_MODE] = CF_RAWPY_MODE_PYTHON
        os.environ[CF_RAWPY_PY_APIPTR] = str(ctypes.pythonapi._handle)
    else:
        raise Exception("Unknown mode: " + (os.getenv(CF_RAWPY_MODE) or "<unset>"))

    jl_exepath = os.getenv(CF_RAWPY_JL_EXE)

    if not jl_exepath:
        raise Exception(f"environment varibale {CF_RAWPY_JL_EXE} is not set")

    jl_opts = shlex.split(os.getenv(CF_RAWPY_JL_OPTS, ""))
    cmd = [jl_exepath, *jl_opts, '--startup-file=no', '-O0', '--compile=min', '-e', julia_info_query]
    bindir, libpath, sysimage, project_dir = subprocess.run(
        cmd, check=True, capture_output=True, encoding='utf8'
    ).stdout.splitlines()


    old_cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(libpath)))
        lib = ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)
        try:
            init_func = lib.jl_init_with_image
        except AttributeError:
            init_func = lib.jl_init_with_image__threading

        argc, argv = args_from_config(jl_exepath, jl_opts)
        lib.jl_parse_opts(ctypes.pointer(argc), ctypes.pointer(argv))
        
        init_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        init_func.restype = None
        init_func(bindir.encode('utf8'), sysimage.encode('utf8'))
        lib.jl_eval_string.argtypes = [ctypes.c_char_p]
        lib.jl_eval_string.restype = ctypes.c_void_p
        
        if not lib.jl_eval_string(rf"""
        try
            import Pkg
            Pkg.activate({json.dumps(project_dir, ensure_ascii=False)}, io=devnull)
            import RawPython
            RawPython.CPython.init()
        catch err
            showerror(stderr, err, catch_backtrace())
            flush(stderr)
            rethrow()
        end
        """.encode('utf8')):
            raise RuntimeError("invalid julia initialization")


        def _eval_jl(x: str):
            return lib.jl_eval_string(x.encode('utf-8'))

    finally:
        os.chdir(old_cwd)
