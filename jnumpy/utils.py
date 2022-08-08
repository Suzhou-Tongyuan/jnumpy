import os

def set_julia_path(path: str):
    path = os.path.abspath(os.path.expanduser(path))
    os.environ["TYPY_JL_EXE"] = path

def set_julia_mirror(upstream: str):
    os.environ["JILL_UPSTREAM"] = upstream