from __future__ import annotations
from jnumpy.envars import TyPython_directory
from jnumpy.utils import escape_to_julia_rawstr
from jnumpy.init import JuliaError, exec_julia
from dataclasses import dataclass
import typing_extensions
import contextlib
import pathlib
import typing
import pydantic
import toml
import os

_T = typing.TypeVar("_T")
_G = typing.TypeVar("_G")


class JuliaProjectDict(typing_extensions.TypedDict):
    name: str
    deps: typing.Dict[str, str]


@pydantic.dataclasses.dataclass
class JuliaProject:
    name: str
    deps: typing.Dict[str, str]


class ParseProjectResult(typing.Generic[_T, _G]):
    proj: _T
    fullproj: _G


def parse_project(
    result: ParseProjectResult,
    path: pathlib.Path,
) -> typing_extensions.TypeGuard[ParseProjectResult[JuliaProject, JuliaProjectDict]]:
    try:
        fullproj = toml.load(path)
    except toml.TomlDecodeError:
        return False
    try:
        proj: JuliaProject = JuliaProject.__pydantic_model__.parse_obj(fullproj)  # type: ignore
        result.proj = proj
        result.fullproj = fullproj
        return True
    except pydantic.ValidationError:
        return False


def include_src(src_file: str, current_file_path: str = "./__init__.py"):
    """
    include julia module in src_file
    Arguments:
      src_file:
        the path of julia file releative to file path.
      file_path(option):
        should be `__file__`, empty in repl mode.
    """
    # activate project before include module
    project_dir = pathlib.Path(current_file_path).absolute().parent
    src_abspath = project_dir.joinpath(src_file)
    from jnumpy import exec_julia

    exec_julia(r"include({})".format(escape_to_julia_rawstr(src_abspath.as_posix())))


@dataclass
class LocalProjectCache:
    julia_module_name: str
    py_package_rootdir: str
    project_path: str
    fullproj: JuliaProjectDict

    def dump_project_(self):
        with open(self.project_path, "w", encoding="utf-8") as f:
            toml.dump(self.fullproj, f)


_rootpath_caches: dict[str, LocalProjectCache] = {}


def load_project(package_entry_filepath: str = "./__init__.py"):
    """
    include the julia module in the current project

    Arguments:
      package_entry_filepath(option):
        should be the `__file__` of the toplevel `__init__.py`.
    """
    # activate project before include module
    py_package_rootdir = pathlib.Path(package_entry_filepath).absolute().parent
    cache = _rootpath_caches.get(py_package_rootdir.as_posix())
    if cache is not None:
        return cache

    project_path = py_package_rootdir.joinpath("Project.toml")
    parse_result = ParseProjectResult()
    if not parse_project(parse_result, project_path):
        raise RuntimeError(
            f"{py_package_rootdir} does not have a Project.toml with a top-level"
            f"entry 'name = xxx' and the '[deps]' section."
        )
    proj = parse_result.proj
    fullproj = parse_result.fullproj
    py_package_rootdir_as_posix = py_package_rootdir.as_posix()
    cache = _rootpath_caches[py_package_rootdir_as_posix] = LocalProjectCache(
        proj.name, py_package_rootdir_as_posix, project_path.as_posix(), fullproj
    )

    from jnumpy import exec_julia

    with activate_project(cache):
        exec_julia("import {0}".format(fullproj["name"]))

    return cache


@contextlib.contextmanager
def activate_project(cache: LocalProjectCache):
    from jnumpy.init import default_project_dir

    _activate_project_impl(cache.py_package_rootdir)

    if not is_initialized():
        exec_julia(f"Pkg.develop(path={escape_to_julia_rawstr(TyPython_directory)})", use_gil=False)
        exec_julia("Pkg.resolve()", use_gil=False)
        exec_julia("Pkg.instantiate()", use_gil=False)
    try:
        yield
    finally:
        _activate_project_impl(default_project_dir)


def init_project(package_entry_filepath: str):
    cache = load_project(package_entry_filepath)
    from jnumpy import exec_julia

    with activate_project(cache):
        exec_julia(
            "import {0};TyPython.CPython.init();{0}.init()".format(
                cache.julia_module_name
            )
        )


def is_initialized():
    try:
        exec_julia(f"Pkg.instantiate(io=devnull)")
        return True
    except JuliaError:
        return False


def _activate_project_impl(project_dir: str | None = None):
    from jnumpy.init import default_project_dir

    project_dir_assure_str = project_dir or default_project_dir
    exec_julia(
        f"Pkg.activate({escape_to_julia_rawstr(project_dir_assure_str)}, io=devnull)"
    )


def add_deps(package_entry_filepath: str):
    # parse the Project.toml in file's dir and add dependencies to working project
    toml_path = os.path.join(
        os.path.dirname(os.path.abspath(package_entry_filepath)), "Project.toml"
    )
    exec_julia(f"TyPython.Utils.add_deps({escape_to_julia_rawstr(toml_path)})")
