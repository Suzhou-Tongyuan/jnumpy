from __future__ import annotations
import io
import os
import sys
import subprocess
import ctypes
import shlex
import contextlib
from .utils import escape_to_julia_rawstr
from .defaults import setup_julia_exe_, get_project_args
from .envars import (
    CF_TYPY_JL_OPTS,
    CF_TYPY_MODE,
    CF_TYPY_MODE_JULIA,
    CF_TYPY_MODE_PYTHON,
    CF_TYPY_PY_APIPTR,
    TyPython_directory,
    InitTools_path,
    SessionCtx,
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
    try
        Base.@eval begin
            __PY_GIL = TyPython.CPython.GIL_BEGIN()
            try
                {}
            finally
                TyPython.CPython.GIL_END(__PY_GIL)
            end
        end
    catch e
        TyPython.CPython.WITH_GIL() do
            errmsg = TyPython.CPython.Utils.capture_out() do
                Base.showerror(stderr, e, catch_backtrace())
            end
            sys = TyPython.CPython.py_import("sys")
            err_o = TyPython.CPython.py_cast(TyPython.CPython.Py, errmsg)
            sys.stderr.write(err_o)
            rethrow()
        end
    end
end
"""

no_gil_template = r"""
begin
    {}
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
    try:
        _eval_jl(x, use_gil)  # type: ignore
    except NameError:
        raise RuntimeError(
            "name '_eval_jl' is not defined, should call init_jl() first"
        )


class JuliaError(Exception):
    pass


def init_jl():
    global _eval_jl
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

    setup_julia_exe_()
    jl_opts = shlex.split(os.getenv(CF_TYPY_JL_OPTS, ""))
    jl_opts_proj = get_project_args()
    cmd = [
        SessionCtx.JULIA_EXE,
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
    SessionCtx.JULIA_START_OPTIONS = [jl_opts_proj, *jl_opts]
    SessionCtx.DEFAULT_PROJECT_DIR = default_project_dir

    old_cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(libpath)))
        lib = ctypes.PyDLL(libpath, mode=ctypes.RTLD_GLOBAL)
        try:
            init_func = lib.jl_init_with_image
        except AttributeError:
            init_func = lib.jl_init_with_image__threading

        argc, argv = args_from_config(SessionCtx.JULIA_EXE, SessionCtx.JULIA_START_OPTIONS)
        lib.jl_parse_opts(ctypes.pointer(argc), ctypes.pointer(argv))

        init_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        init_func.restype = None
        init_func(bindir.encode("utf8"), sysimage.encode("utf8"))
        lib.jl_eval_string.argtypes = [ctypes.c_char_p]
        lib.jl_eval_string.restype = ctypes.c_void_p
        lib.jl_exception_clear.restype = None

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
                    lib.jl_exception_clear()
                    raise JuliaError(ef.getvalue())
                return None

        exec_julia(
            f"""
            import Pkg
            Pkg.activate({escape_to_julia_rawstr(default_project_dir)}, io=devnull)
            include({escape_to_julia_rawstr(InitTools_path)})
        """,
            use_gil=False,
        )

        try:
            exec_julia("import TyPython", use_gil=False)
        except JuliaError:
            try:
                exec_julia(
                    f"InitTools.setup_environment({escape_to_julia_rawstr(TyPython_directory)})",
                    use_gil=False,
                )
            except JuliaError:
                pass
            exec_julia(
                f"InitTools.force_resolve({escape_to_julia_rawstr(TyPython_directory)})",
                use_gil=False,
            )
        try:
            exec_julia(
                rf"""
                import TyPython
                import TyPython.CPython
                TyPython.CPython.init()
            """,
                use_gil=False,
            )
        except JuliaError:
            raise RuntimeError("invalid julia initialization")

    finally:
        os.chdir(old_cwd)
