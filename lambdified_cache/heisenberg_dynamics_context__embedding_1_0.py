import numpy as np

def heisenberg_dynamics_context__embedding_1_0 (X):
    retval = np.ndarray((2, 3), dtype=float)
    retval[0,0] = 1
    retval[0,1] = 0
    retval[0,2] = 0
    retval[1,0] = 0
    retval[1,1] = X[0]
    retval[1,2] = -2*X[0] - 1/np.sqrt(np.pi)
    return retval

