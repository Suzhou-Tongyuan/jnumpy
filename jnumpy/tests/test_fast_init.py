def test_fast_init():
    import jnumpy as np
    np.init_jl(experimental_fast_init=True)
    np.exec_julia("println(:success)")
    return