# JNumPy: writing high-performance C extensions for Python in minutes

## Install JNumPy

Requirements:

- Python >= 3.7

You can install the Python package `jnumpy` with the following command:

`pip install julia-numpy`.

Note that JNumPy will install julia in `JNUMPY_HOME` for you, if there is no Julia installation available.

## Quick Start

1. add the `TyPython` dependency

    ```
    julia --project=. -e "import Pkg; Pkg.add(\"TyPython\")"
    ```

2. write and export julia functions in file `example.jl`

    ```julia
    module example

    using TyPython
    using TyPython.CPython
    CPython.init()

    @export_py function mat_mul(a::StridedArray, b::StridedArray)::StridedArray
        return a * b
    end

    function init()
        @export_pymodule example begin
            jl_mat_mul = Pyfunc(jl_mat_mul)
        end
    end
    ```

3. initialize and import julia function in python

    ```python
    from jnumpy import init_jl, exec_julia, include_src
    import jnumpy as np
    init_jl()
    include_src("example.jl", __file__)
    exec_julia("example.init()")

    from example import jl_mat_mul

    x = np.array([[1,2],[3,4]])
    y = np.array([[4,3],[2,1]])
    jl_mat_mul(x, y)
    # array([[ 8,  5],
    #       [20, 13]])
    ```


## Environment Variables

- `JNUMPY_HOME`:

    The home directory for JNumPy-specific settings. The default value is `~/.jnumpy`. JNumPy runs julia in a default environment (`$JNUMPY_HOME/envs/default`). In case that you don't have a julia executable, JNumPy installs julia into `$JNUMPY_HOME` using [jill.py](https://github.com/johnnychen94/jill.py).

- `TYPY_JL_EXE`:

    The path of the julia executable in use.

- `TYPY_JL_OPTS`:

    Command-line options when launching julia. If you want to use a custom environment, you could set `--project=<dir>`. `TYPY_JL_OPTS` is the same as those arguments passed to `julia`.

## Examples

There are several examples presented in the `demo` directory. Those examples are standalone Python packages created using JNumPy, and can be imported if you have JNumPy installed.

- `demo/basic`: a tiny Python package to give an example of how to use JNumPy.

- `demo/kmeans`: a tiny Python package wrapping [ParallelKMeans.jl](https://pydatablog.github.io/ParallelKMeans.jl/stable/). It produces a 10x performance gain against Scikit-Learn.

- `demo/fft`: a tiny Python package wrapping FFTW.jl. It allows users to access FFT plans for accelerating FFTs.

## Contributions

Open-source contributions are kindly accepted and appreciated including bug reports, documentations, pull requests, and general suggestions.
