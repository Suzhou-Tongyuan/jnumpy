from __future__ import annotations
import os
import io
import json
import subprocess
import contextlib


def set_julia_path(path: str):
    path = os.path.abspath(os.path.expanduser(path))
    os.environ["TYPY_JL_EXE"] = path


def set_julia_mirror(upstream: str):
    os.environ["JILL_UPSTREAM"] = upstream


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
