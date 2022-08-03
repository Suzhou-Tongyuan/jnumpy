# JNumpy

## Writing Python C extensions in Julia



### Install JNumpy

`pip install jnumpy`

### Quick Start

1. add TyPython dependency
```
julia --project=. -e "import Pkg; Pkg.add("TyPython")
```

2. write julia functions and export in file `example.jl`
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

3. init and import julia's function in python
```python
from jnumpy import init_jl, exec_julia, include_src
import jnumpy as np
init_jl()
include_src(__file__, "example.jl")
exec_julia("example.init()")

from example import jl_mat_mul

x = np.array([[1,2],[3,4]])
y = np.array([[3,4],[2,1]])
jl_mat_mul(x, y)
# array([[ 8,  5],
#       [20, 13]])
```


### Environment Variable




