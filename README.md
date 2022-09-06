# JNumPy: writing high-performance C extensions for Python in minutes

## Install JNumPy

Requirements:

- Python >= 3.7

You can install the Python package `jnumpy` with the following command:

`pip install julia-numpy`.

Note that JNumPy will install julia in `JNUMPY_HOME` for you, if there is no Julia installation available.

## Usage

1. create a Python package `example`, write and export julia functions in the file `example/src/example.jl`

    ```julia
    module example

    using TyPython
    using TyPython.CPython

    @export_py function mat_mul(a::AbstractArray, b::AbstractArray)::Array
        return a * b
    end

    function init()
        @export_pymodule _example begin
            jl_mat_mul = Pyfunc(mat_mul)
        end
    end

    # the following code is optional,
    # but makes Python code loading much faster since the second time.
    precompile(init, ())

    end
    ```

2. create `example/Project.toml` as follows:

    ```toml
    name = "example"  # this is required to find the julia's entry module

    [deps]
    # specify your julia dependencies here
    ```

3. initialize and import the julia functions at `example/__init__.py`:

    ```python
    import jnumpy as np
    # you may call np.set_julia_mirror(server) to set the julia package server,
    # or leave the argument server empty to automatically search the nearest mirror.
    np.init_jl()
    np.init_project(__file__)

    from _example import jl_mat_mul

    __all__ = ['jl_mat_mul']
    ```

4. enjoy your Python extension package:


    This is the structure of your Python extension package:

    ```bash
    > ls -R
    example/:
        __init__.py  Project.toml  src

    example/src:
        example.jl
    ```

    This is how you use it:

    ```python
    from example import jl_mat_mul
    x = np.array([[1,2],[3,4]])
    y = np.array([[4,3],[2,1]])
    jl_mat_mul(x, y)
    # array([[ 8,  5],
    #       [20, 13]])
    ```

## Environment Variables

- `JNUMPY_HOME`:

    The home directory for JNumPy-specific settings. The default value is `~/.jnumpy`. In case that you don't have a julia executable, JNumPy installs julia into `$JNUMPY_HOME` using [jill.py](https://github.com/johnnychen94/jill.py).

- `TYPY_JL_EXE`:

    The path of the julia executable in use.

- `TYPY_JL_OPTS`:

    Command-line options when launching julia. JNumPy runs julia in a default environment (`path/to/package/jnumpy/JNumPyEnv`). If you want to use a custom environment, you could set `--project=<dir>`. `TYPY_JL_OPTS` is the same as those arguments passed to `julia`.

## Examples

There are several examples presented in the `demo` directory. Those examples are standalone Python packages created using JNumPy, and can be imported if you have JNumPy installed.

- `demo/basic`: a tiny Python package to give an example of how to use JNumPy.

- `demo/kmeans`: a tiny Python package wrapping [ParallelKMeans.jl](https://pydatablog.github.io/ParallelKMeans.jl/stable/). It produces a 10x performance gain against Scikit-Learn.

- `demo/fft`: a tiny Python package wrapping [FFTW.jl](https://github.com/JuliaMath/FFTW.jl), and indirectly the GPL-licenced FFTW library. It allows users to access FFT plans for accelerating FFTs.

## Contributions

Open-source contributions are kindly accepted and appreciated including bug reports, documentations, pull requests, and general suggestions.
