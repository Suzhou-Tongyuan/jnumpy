from extension import jl_not, int_add, float_add, complex_mul_two, str_concat, tuple_return, mat_mul, set_zero, jl_fft
import numpy as np


def test_not():
    assert jl_not(False)

def test_add():
    assert int_add(1, 2) == 3
    np.testing.assert_almost_equal(float_add(1.0, 2.0), 3.0)

def test_complex():
    x = complex(1.0, 2.1)
    np.testing.assert_almost_equal(complex_mul_two(x), 2*x)

def test_str_concat():
    assert str_concat("a", "b") == "ab"

def test_tuple_return():
    assert tuple_return(1, "a") == (1, "a")

def test_mat_mul():
    x = np.random.rand(2,2)
    y = np.random.rand(2,2)
    actual = mat_mul(x, y)
    desired = x @ y
    np.testing.assert_array_equal(actual, desired)

def test_set_zero():
    x = np.random.rand(2)
    set_zero(x)
    np.testing.assert_almost_equal(x[0], 0.0)

def test_fft():
    x = np.exp(2j * np.pi * np.arange(8) / 8)
    actual = jl_fft(x)
    desired = np.fft.fft(x)
    np.testing.assert_array_almost_equal(actual, desired)
