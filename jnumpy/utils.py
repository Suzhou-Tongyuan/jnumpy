from __future__ import annotations
import os
import io
import json
import platform
import subprocess
import contextlib
import time
import http.client
import threading
from .envars import SessionCtx
import pathlib
import argparse
from .envars import DefaultProj_directory


def set_julia_path(path: str):
    """
    Set the environment `TYPY_JL_EXE`.
    Arguments:
      path:
        path of the julia executable.
    """
    path = os.path.abspath(os.path.expanduser(path))
    os.environ["TYPY_JL_EXE"] = path


def set_julia_mirror(server: str = ""):
    """
    Set the environment `JULIA_PKG_SERVER` for julia package server.
    Only work when users haven't set the environment `JULIA_PKG_SERVER` manually.
    Arguments:
      server:
        the url of julia package server, like "https://pkg.julialang.org",
        or leave it empty to auto search a nearest server mirror.
    """
    if not os.environ.get("JULIA_PKG_SERVER"):
        server = server if server else get_fast_mirror()
        os.environ["JULIA_PKG_SERVER"] = server


# python version of https://github.com/johnnychen94/PkgServerClient.jl/blob/master/src/PkgServerClient.jl
def get_fast_mirror():
    response_time = registry_response_time()
    fast_mirror = sorted(response_time.items(), key=lambda x: x[1])[0][0]
    # When all fails to response in a very limited `timeout` time, set the default to "JuliaLang"
    if response_time[fast_mirror] >= 1000:
        url = "https://pkg.julialang.org"
    else:
        url = f"https://{fast_mirror}/julia"
    return url


def registry_response_time(timeout: float = 1.0):
    timeout = max(
        0.001, timeout
    )  # The minimal is 1ms, otherwise it is likely to be ignored.

    hosts = [
        "mirrors.bfsu.edu.cn",
        "mirror.iscas.ac.cn",
        "mirrors.nju.edu.cn",
        "opentuna.cn",
        "mirrors.sjtug.sjtu.edu.cn",
        "mirrors.sustech.edu.cn",
        "mirrors.tuna.tsinghua.edu.cn",
        "mirrors.ustc.edu.cn",
    ]
    num_threads = len(hosts)
    _registry_response_time = {key: float("inf") for key in hosts}

    def response_time(conn: http.client.HTTPSConnection, host: str):
        start = time.time()
        try:
            conn.request("HEAD", "/julia/registries")
            conn.getresponse()
            _registry_response_time[host] = time.time() - start
        except:
            _registry_response_time[host] = float("inf")

    def wait_thread(host: str, timeout: float):
        conn = http.client.HTTPSConnection(host)
        worker = threading.Thread(target=response_time, args=(conn, host))
        worker.start()
        worker.join(timeout)
        conn.close()

    workers = []
    for i in range(num_threads):
        worker = threading.Thread(target=wait_thread, args=(hosts[i], timeout))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

    return _registry_response_time


def escape_string(s: str):
    return json.dumps(s, ensure_ascii=False)


def escape_to_julia_rawstr(s: str):
    return "raw" + escape_string(s)


def invoke_interpreted_julia(jl_exepath: str, args: list[str], *, suppress_error=False):
    if suppress_error:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return subprocess.check_output(
                    [
                        jl_exepath,
                        "--startup-file=no",
                        "-O0",
                        "--compile=min",
                        *args,
                    ]
                )
        except subprocess.CalledProcessError:
            return None
    else:
        return subprocess.check_output(
            [
                jl_exepath,
                "--startup-file=no",
                "-O0",
                "--compile=min",
                *args,
            ]
        )


project_parser = argparse.ArgumentParser()
project_parser.add_argument("--project", type=str, default=None)
sysimage_parser = argparse.ArgumentParser()
sysimage_parser.add_argument("-J", "--sysimage", type=str, default=None)


def try_search_libjulia():
    if platform.system().lower() == "windows":
        libjulia = (
            pathlib.Path(SessionCtx.JULIA_EXE)
            .parent.absolute()
            .joinpath("libjulia.dll")
        )
    elif platform.system().lower() == "linux":
        libjulia = (
            pathlib.Path(SessionCtx.JULIA_EXE)
            .parent.parent.absolute()
            .joinpath("lib", "libjulia.so")
        )
    elif platform.system().lower() == "darwin":
        libjulia = (
            pathlib.Path(SessionCtx.JULIA_EXE)
            .parent.parent.absolute()
            .joinpath("lib", "julia", "libjulia.dylib")
        )
    else:
        raise NotImplementedError(
            f"Unsupported platform {platform.system()} in fast init mode"
        )

    if libjulia.is_file():
        return libjulia.as_posix()
    else:
        raise FileNotFoundError(f"libjulia not found at {libjulia}")
    return


def try_search_sysimage():
    if platform.system().lower() == "windows":
        sysimage = (
            pathlib.Path(SessionCtx.JULIA_EXE)
            .parent.parent.absolute()
            .joinpath("lib", "julia", "sys.dll")
        )
    elif platform.system().lower() == "linux":
        sysimage = (
            pathlib.Path(SessionCtx.JULIA_EXE)
            .parent.parent.absolute()
            .joinpath("lib", "julia", "sys.so")
        )
    elif platform.system().lower() == "darwin":
        sysimage = (
            pathlib.Path(SessionCtx.JULIA_EXE)
            .parent.parent.absolute()
            .joinpath("lib", "julia", "sys.dylib")
        )
    else:
        raise NotImplementedError(
            f"Unsupported platform {platform.system()} in fast init mode"
        )

    if sysimage.is_file():
        return sysimage.as_posix()
    else:
        raise FileNotFoundError(f"sysimage not found at {sysimage}")
    return


def guess_julia_init_params(args: list[str]):
    bindir = pathlib.Path(SessionCtx.JULIA_EXE).parent.absolute().as_posix()
    project_opts, _ = project_parser.parse_known_args(args)
    if project_opts.project:
        default_project_dir = pathlib.Path(project_opts.project).absolute().as_posix()
    else:
        default_project_dir = DefaultProj_directory
    try:
        libjulia = try_search_libjulia()
        sysimage = try_search_sysimage()
    except:
        return None
    return (bindir, libjulia, sysimage, default_project_dir)
