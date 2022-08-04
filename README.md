# JNumpy

## Writing Python C extensions in Julia



### Install JNumpy

`pip install jnumpy`

### Quick Start

1. add TyPython dependency
```
julia --project=. -e "import Pkg; Pkg.add("TyPython")
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

3. init and import julia function in python
```python
from jnumpy import init_jl, exec_julia, include_src
import jnumpy as np
init_jl()
include_src("example.jl", __file__)
exec_julia("example.init()")

from example import jl_mat_mul

x = np.array([[1,2],[3,4]])
y = np.array([[3,4],[2,1]])
jl_mat_mul(x, y)
# array([[ 8,  5],
#       [20, 13]])
```


### Environment Variable

`JNUMPY_HOME`: The home dir of jnumpy. The default value is `~/.jnumpy`. jnumpy run julia in a default environment `$JNUMPY_HOME/envs/default`. If you don't have a julia, jnumpy could install julia in `$JNUMPY_HOME` by [JILL](https://github.com/johnnychen94/jill.py)
`RAWPY_JL_EXE`: The julia Executable Path. 
`RAWPY_JL_OPTS`: Command-line options when launching julia. If you want to use a custom environment, you could set `--project=<dir>`


