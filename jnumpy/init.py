from __future__ import annotations
import io
import os
import subprocess
import ctypes
import json
import shlex
import contextlib
import pathlib
import typing_extensions
import toml
import typing
from pydantic.dataclasses import dataclass
from pydantic import ValidationError
from .defaults import get_julia_exe, get_project_args

CF_TYPY_MODE = "TYPY_MODE"
CF_TYPY_PY_APIPTR = "TYPY_PY_APIPTR"
CF_TYPY_PY_DLL = "TYPY_PY_DLL"
CF_TYPY_MODE_PYTHON = "PYTHON-BASED"
CF_TYPY_MODE_JULIA = "JULIA-BASED"
CF_TYPY_JL_EXE = "TYPY_JL_EXE"
CF_TYPY_JL_OPTS = "TYPY_JL_OPTS"
CF_JNUMPY_HOME = "JNUMPY_HOME"

# debug
# os.environ[CF_TYPY_JL_OPTS] = "--compile=min -O0"

julia_info_query = r"""
import Libdl
import Pkg
println(Base.Sys.BINDIR)
println(abspath(Libdl.dlpath("libjulia")))
println(unsafe_string(Base.JLOptions().image_file))
println(dirname(Pkg.project().path))
""".replace('\n', ';').replace('\r', ';')

TyPython_dir = os.path.abspath("./TyPython")

class JuliaProjectDict(typing_extensions.TypedDict):
    name: str
    deps: typing.Dict[str, str]

@dataclass
class JuliaProject:
    name: str
    deps: typing.Dict[str, str]

class ParseProjectResult:
    proj: JuliaProject

def check_project(fullproj: dict, result: ParseProjectResult) -> typing_extensions.TypeGuard[JuliaProjectDict]:
    try:
        proj: JuliaProject = JuliaProject.__pydantic_model__.parse_obj(fullproj) # type: ignore
        result.proj = proj
        return True
    except ValidationError:
        return False

exec_template = r"""
begin
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


def args_from_config(exepath: str, args: list):
    args = [exepath] + args

    # python 2 is deprecated, we just consider python 3
    argv: list[bytes] = [arg.encode('utf-8') for arg in args]
    argc = len(argv)
    argc = ctypes.c_int(argc)
    argv = ctypes.POINTER(ctypes.c_char_p)((ctypes.c_char_p * len(argv))(*argv))  # type: ignore
    return argc, argv

def escape_string(s: str):
    return json.dumps(s, ensure_ascii=False)

def exec_julia(x):
    global _eval_jl
    _eval_jl(x)  # type: ignore

def include_src(src_file: str, file_path: str=""):
    """
    include julia module in src_file
    Arguments:
      src_file:
        the path of julia file releative to file path.
      file_path(option):
        should be `__file__`, empty in repl mode.
    """
    # activate project before include module
    file_dir = os.path.dirname(file_path)
    src_path = os.path.join(file_dir, src_file)
    src_path = os.path.abspath(src_path)
    activate_proj(file_dir)
    exec_julia("include({})".format(escape_string(src_path)))
    activate_proj(default_project_dir)

_path_to_modulenames: dict[str, tuple[str, str, str, JuliaProjectDict]] = {}

def load_project(file_path: str="."):
    """
    include julia module in project

    Arguments:
      file_path(option):
        should be `__file__`, empty in repl mode.
    """
    # activate project before include module
    file_dir = pathlib.Path(file_path).absolute().parent.as_posix()
    res = _path_to_modulenames.get(file_dir)
    if res is not None:
        return res

    project_path = os.path.join(file_dir, "Project.toml")
    fullproj = toml.load(str(project_path))
    result = ParseProjectResult()
    if not check_project(fullproj, result):
        raise RuntimeError(f"{file_dir} does not have a Project.toml with a top-level"
                           f"entry 'name = xxx' and the '[deps]' section.")
    proj = result.proj
    res = _path_to_modulenames[file_dir] = (proj.name, file_dir, project_path, fullproj)
    with activate_project(project_path, file_dir, fullproj):
        exec_julia("import {0}".format(fullproj["name"]))

    return res

@contextlib.contextmanager
def activate_project(project_path: str, file_dir: str, fullproj: JuliaProjectDict):
    if fullproj["deps"].pop("TyPython", None):
        with open(project_path, 'w', encoding='utf-8') as f:
                toml.dump(fullproj, f)
    activate_proj(file_dir)
    try:
        fullproj['deps']['TyPython'] = "9c4566a2-237d-4c69-9a5e-9d27b7d0881b"
        with open(project_path, 'w', encoding='utf-8') as f:
            toml.dump(fullproj, f)
        try:
            yield
        finally:
            with open(project_path, 'w', encoding='utf-8') as f:
                del fullproj['deps']["TyPython"]
                toml.dump(fullproj, f)
    finally:
        activate_proj(default_project_dir)

def init_project(file_path):
    modulename, file_dir, project_path, fullproj = load_project(file_path)
    with activate_project(project_path, file_dir, fullproj):
        exec_julia("import {0};TyPython.CPython.init();{0}.init()".format(modulename))

def activate_proj(proj_dir: str):
    global _activate_proj
    _activate_proj(proj_dir)

def add_deps(file_path: str):
    # parse the Project.toml in file's dir and add dependencies to working project
    toml_path = os.path.join(os.path.dirname(file_path), "Project.toml")
    global _add_deps
    _add_deps(toml_path)

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
    cmd = [jl_exepath, jl_opts_proj, *jl_opts, '--startup-file=no', '-O0', '--compile=min', '-e', julia_info_query]
    bindir, libpath, sysimage, default_project_dir = subprocess.run(
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
            Pkg.activate({escape_string(default_project_dir)}, io=devnull)
            if !haskey(Pkg.project().dependencies, "TyPython")
                Pkg.develop(path={escape_string(TyPython_dir)})
            end
            Pkg.instantiate()
            import TyPython
            TyPython.CPython.init()
        catch err
            showerror(stderr, err, catch_backtrace())
            rethrow()
        end
        """.encode('utf8')):
            raise RuntimeError("invalid julia initialization")


        def _eval_jl(x: str):
            with contextlib.redirect_stderr(io.StringIO()) as ef:
                source_code = exec_template.format(x).encode('utf8')
                if not lib.jl_eval_string(source_code) or lib.jl_exception_occurred():
                    raise JuliaError(ef.getvalue())
                return None

        def _activate_proj(proj_dir: str):
            if not lib.jl_eval_string(rf"""
            try
                Pkg.activate({escape_string(proj_dir)}, io=devnull)
                Pkg.instantiate()
            catch err
                showerror(stdout, err, catch_backtrace())
                rethrow()
            end
            """.encode('utf8')):
                raise RuntimeError(f"fail to activate julia projects {proj_dir}")

        def _add_deps(toml_path: str):
            if not lib.jl_eval_string(rf"""
            try
                TyPython.Utils.add_deps({escape_string(toml_path)})
            catch err
                showerror(stderr, err, catch_backtrace())
                rethrow()
            end
            """.encode('utf8')):
                raise RuntimeError("fail to add julia dependencies")
    finally:
        os.chdir(old_cwd)
