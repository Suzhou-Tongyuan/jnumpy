import jnumpy as np
import numpy.testing as nptest
import pytest

def test_all():
    np.init_jl()
    np.exec_julia(
        r"CPython.init_jlwrap()",
        use_gil=True,
    )

    f = np.jl_eval("f(a,b) = a*b") # type: ignore
    g = np.jl_eval("g(a) = begin a[1]=0.0 end") # type: ignore
    Point = np.jl_eval("struct Point x end; Point") # type: ignore

    a = np.random.rand(10,10)
    b = np.random.rand(10,10)
    out = f(a, b)
    nptest.assert_array_almost_equal(a @ b, out)
    g(a)
    assert a[0,0] == 0.0
    with pytest.raises(TypeError):
        g()
    c = Point(1)
    assert c.x == 1


