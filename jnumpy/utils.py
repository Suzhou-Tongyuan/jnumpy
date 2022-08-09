import os
import json


def set_julia_path(path: str):
    path = os.path.abspath(os.path.expanduser(path))
    os.environ["TYPY_JL_EXE"] = path


def set_julia_mirror(upstream: str):
    os.environ["JILL_UPSTREAM"] = upstream


def escape_string(s: str):
    return json.dumps(s, ensure_ascii=False)


def escape_to_julia_rawstr(s: str):
    return "raw" + escape_string(s)
