# TyPython

TyPython is a Julia package to work with Python in Julia and focusing on the following features:

- the capability of building portable system images.

- high performance bidirectional interaction between Python and Julia. For instance. functions wrapped by TyPython can avoid dynamic dispatch.

- creating Python packages in Julia, i.e., "Python C Extensions".

- deterministic bidirectional data transformation/sharing between Python and Julia:

    In the Julia side, NO automatic conversions are performed. Conversions are classified into casting and coercion: giving a python int `o`, `py_coerce(Cdouble, o)` fails, while `py_cast(Cdouble, o)` succeeds.

    Python side can use functions wrapped from Julia. A wrapped function accepts only Python arguments of the following types:

    - `int`
    - `bool`
    - `float`
    - `complex`
    - `str`
    - `np.ndarray`
    - tuples of above types



```julia
# export TYPY_PY_DLL="/path/to/python/so"
using TyPython
using TyPython.CPython
CPython.init()
py_builtin = CPython.get_py_builtin()
x = py_builtin.dict()
# Py({})

x[py_cast(Py, 1)] = py_cast(Py, nothing)
x
# Py({1: None})
```