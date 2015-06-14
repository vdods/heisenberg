import numpy as np
import sympy

def tensor (name, shape):
    return np.array(sympy.symbols(name+''.join('(0:{0})'.format(s) for s in shape))).reshape(shape)

