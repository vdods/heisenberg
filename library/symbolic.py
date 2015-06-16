import numpy as np
import sympy

def variable (name):
    return sympy.symbols(name)

def tensor (name, shape):
    return np.array(sympy.symbols(name+''.join('(0:{0})'.format(s) for s in shape))).reshape(shape)

