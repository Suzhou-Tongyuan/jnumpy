from jnumpy.tests.extension import (
    jl_not,
    int_add,
    float_add,
    complex_mul_two,
    str_concat,
    tuple_return,
    mat_mul,
    set_zero,
    jl_fft,
)
import os
import subprocess
import numpy as np
import pytest

dtype_list = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


def test_not():
    assert jl_not(False)


def test_add():
    assert int_add(1, 2) == 3
    np.testing.assert_almost_equal(float_add(1.0, 2.0), 3.0)


def test_complex():
    x = complex(1.0, 2.1)
    np.testing.assert_almost_equal(complex_mul_two(x), 2 * x)


def test_str_concat():
    assert str_concat("a", "b") == "ab"


def test_tuple_return():
    assert tuple_return(1, "a") == (1, "a")


@pytest.mark.parametrize("dtype", dtype_list)
def test_mat_mul(dtype):
    x = np.asarray([[1.2, 3.4], [2.3, 5.6]], dtype=dtype)
    y = np.asarray([[7.8, 5e-3], [6.75, 8.234]], dtype=dtype)
    actual = mat_mul(x, y)
    desired = x @ y
    assert actual.dtype == desired.dtype
    np.testing.assert_array_almost_equal(actual, desired, decimal=5)

def test_bool_array():
    x = np.asarray([[True, False], [True, False]])
    x_int = x.astype('i')
    actual = mat_mul(x, x)
    desired = x_int @ x_int
    assert np.all(actual == desired)

def test_set_zero():
    x = np.random.rand(2)
    set_zero(x)
    np.testing.assert_almost_equal(x[0], 0.0)


def test_fft():
    x = np.exp(2j * np.pi * np.arange(8) / 8)
    actual = jl_fft(x)
    desired = np.fft.fft(x)
    np.testing.assert_array_almost_equal(actual, desired)

def test_subprocess():
    assert os.getenv("TYPY_PY_APIPTR")
    cmd = [
        "python",
        "-c",
        "import jnumpy as np; np.init_jl(); np.exec_julia(\"print(1)\")"
    ]
    out = subprocess.run(
        cmd, check=True, capture_output=True, encoding="utf8"
    )
    assert out.stdout == "1"
