from __future__ import annotations
import io
import os
import subprocess
import ctypes
import shlex
import contextlib
from .utils import escape_to_julia_rawstr
from .defaults import get_julia_exe, get_project_args
from .envars import (
    CF_TYPY_JL_OPTS,
    CF_TYPY_MODE,
    CF_TYPY_MODE_JULIA,
    CF_TYPY_MODE_PYTHON,
    CF_TYPY_PY_APIPTR,
    TyPython_directory,
)

# XXX: adding an environment variable for fast debugging:
# os.environ[CF_TYPY_JL_OPTS] = "--compile=min -O0"


julia_info_query = r"""
import Libdl
import Pkg
println(Base.Sys.BINDIR)
println(abspath(Libdl.dlpath("libjulia")))
println(unsafe_string(Base.JLOptions().image_file))
println(dirname(Pkg.project().path))
""".replace(
    "\n", ";"
).replace(
    "\r", ";"
)


gil_template = r"""
begin
    import TyPython
    using TyPython.CPython
    try
        Base.@eval begin
            __PY_GIL = CPython.GIL_BEGIN()
            try
                {}
            finally
                CPython.GIL_END(__PY_GIL)
            end
        end
    catch e
        CPython.WITH_GIL() do
            errmsg = CPython.Utils.capture_out() do
                Base.showerror(stderr, e, catch_backtrace())
            end
            sys = CPython.py_import("sys")
            err_o = CPython.py_cast(CPython.Py, errmsg)
            sys.stderr.write(err_o)
            rethrow()
        end
    end
end
"""

no_gil_template = r"""
begin
    import TyPython
    using TyPython.CPython
    try
        {}
    catch e
        showerror(stderr, e, catch_backtrace())
    end
end
"""


def args_from_config(exepath: str, args: list):
    args = [exepath] + args

    # python 2 is deprecated, we just consider python 3
    argv: list[bytes] = [arg.encode("utf-8") for arg in args]
    argc = len(argv)
    argc = ctypes.c_int(argc)
    argv = ctypes.POINTER(ctypes.c_char_p)((ctypes.c_char_p * len(argv))(*argv))  # type: ignore
    return argc, argv


def exec_julia(x, use_gil: bool = True):
    global _eval_jl
    _eval_jl(x, use_gil)  # type: ignore


class JuliaError(Exception):
    pass


def init_jl():
    global _eval_jl
    global _add_deps
    global _activate_proj
    global default_project_dir
    if os.getenv(CF_TYPY_MODE) == CF_TYPY_MODE_JULIA:
        return
    elif os.getenv(CF_TYPY_MODE) == CF_TYPY_MODE_PYTHON:
        assert os.getenv(CF_TYPY_PY_APIPTR, str(ctypes.pythonapi._handle))
        return
    elif not os.getenv(CF_TYPY_MODE):
        os.environ[CF_TYPY_MODE] = CF_TYPY_MODE_PYTHON
        os.environ[CF_TYPY_PY_APIPTR] = str(ctypes.pythonapi._handle)
    else:
        raise Exception("Unknown mode: " + (os.getenv(CF_TYPY_MODE) or "<unset>"))

    jl_exepath = get_julia_exe()

    jl_opts = shlex.split(os.getenv(CF_TYPY_JL_OPTS, ""))
    jl_opts_proj = get_project_args()
    cmd = [
        jl_exepath,
        jl_opts_proj,
        *jl_opts,
        "--startup-file=no",
        "-O0",
        "--compile=min",
        "-e",
        julia_info_query,
    ]
    bindir, libpath, sysimage, default_project_dir = subprocess.run(
        cmd, check=True, capture_output=True, encoding="utf8"
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
        init_func(bindir.encode("utf8"), sysimage.encode("utf8"))
        lib.jl_eval_string.argtypes = [ctypes.c_char_p]
        lib.jl_eval_string.restype = ctypes.c_void_p

        if not lib.jl_eval_string(
            rf"""
        try
            import Pkg
            Pkg.activate({escape_to_julia_rawstr(default_project_dir)}, io=devnull)

            is_instantiated = try
                Pkg.Operations.is_instantiated(Pkg.Types.Context().env)
                true
            catch
                false
            end
            if !haskey(Pkg.project().dependencies, "TyPython") || !is_instantiated
                Pkg.develop(path={escape_to_julia_rawstr(TyPython_directory)})
                Pkg.resolve()
                Pkg.instantiate()
            end

            import TyPython
            TyPython.CPython.init()
        catch err
            showerror(stdout, err, catch_backtrace())
            rethrow()
        end
        """.encode(
                "utf8"
            )
        ):
            raise RuntimeError("invalid julia initialization")

        def _eval_jl(x: str, use_gil: bool):
            with contextlib.redirect_stderr(io.StringIO()) as ef:
                if use_gil:
                    source_code = gil_template.format(x)
                else:
                    source_code = no_gil_template.format(x)
                source_code_bytes = source_code.encode("utf8")
                if (
                    not lib.jl_eval_string(source_code_bytes)
                ) and lib.jl_exception_occurred():
                    raise JuliaError(ef.getvalue())
                return None

    finally:
        os.chdir(old_cwd)
