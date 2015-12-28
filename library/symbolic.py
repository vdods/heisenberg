# NOTE: This changes a/b to produce a floating point approximation of that
# ratio, not the integer quotient.  For integer quotient, use a//b instead.
from __future__ import division

import dis
import itertools
import numpy as np # TEMP
import sympy

# NOTE: This is used so that array, cos, sin, etc are present in this namespace,
# and that fewer surprises occur in lambdify, thereby requiring fewer entries in
# the replacement_d parameter.
from numpy import *

def multiindex_iterator (shape, melt_1_tuple=False):
    """
    Provides a tuple-valued iterator to iterate over all multi-indices with given shape.
    For example, if shape is (2,3), then the iterated sequence is:

        (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).

    If len(shape) is 1 and melt_1_tuple is True (the default is False), then instead of
    returning single-element tuples (0,), (1,), (2,), ..., (n-1,), it returns the plain-old
    integer sequence

        0, 1, 2, ..., n-1.

    Note that if len(shape) is 0, then the iterable sequence will be a single, empty tuple.
    """
    if len(shape) == 1 and melt_1_tuple:
        return xrange(shape[0])
    else:
        return itertools.product(*map(xrange, shape))

def variable (name):
    """
    A convenient frontend to sympy.symbols, except that it escapes commas, so that the single
    variable name can contain a comma.  E.g. 'Y[0,1]'.
    """
    return sympy.symbols(name.replace(',','\,'))

def tensor (name, shape):
    return np.array(sympy.symbols(name+'_'+'_'.join('(0:{0})'.format(s) for s in shape))).reshape(shape)

def differential (F, X):
    m = len(np.shape(F))
    n = len(np.shape(X))

    # Scalar function
    if m == 0:
        # Univariate derivative
        if n == 0:
            return F.diff(X)
        # Multivariate derivative
        else:
            return np.array([F.diff(X[I]) for I in multiindex_iterator(np.shape(X))]).reshape(np.shape(X))

    # Multivariate function
    else:
        # Univariate derivative
        if n == 0:
            return np.array([F[I].diff(X) for I in multiindex_iterator(np.shape(F))]).reshape(np.shape(F))
        # Multivariate derivative
        else:
            retval_shape = tuple(list(np.shape(F))+list(np.shape(X)))
            return np.array([F[I[:m]].diff(X[I[m:]]) for I in multiindex_iterator(retval_shape)]).reshape(retval_shape)

def D (F, *X_v):
    """Compute the iterated differential of F with respect to the elements of the iterable X_v."""
    compiled_function = F
    for X in X_v:
        compiled_function = differential(compiled_function, X)
    return compiled_function

def lambdify (F, X, replacement_d={}, print_stuff=False):
    """
    Return a Python function version of the sybolic m-tensor function F, with respect to the n-tensor variable X.
    Both F and X can be of type np.ndarray.  The length of np.shape(F) is m, whereas the length of np.shape(X) is n.
    F and X can still be scalars as well, they don't have to be tensors.

    This uses eval to generate the code, and the repr of various things like np.array or sympy.cos show up as
    just 'array' and 'cos', so unless you've imported the correct versions of those into your global namespace,
    you'll need to specify what they each map to in replacement_d.  Also, the np.array will have dtype=object
    unless changed explicitly.  For example,

        replacement_d={'array':'np.array', 'dtype=object','dtype=float', 'cos':'np.cos'}

    Note: this uses eval, so it's probably very insecure.
    """

    m = len(np.shape(F))
    n = len(np.shape(X))

    # Function domain is 0-tensor
    if n == 0:
        function_source_code = 'lambda {0}:{1}'.format(repr(X), repr(F))
    # Function domain is 1-tensor or higher and function codomain is 0-tensor and 
    elif m == 0:
        Y = np.array([variable('Y[{0}]'.format(','.join(itertools.imap(str,I)))) for I in multiindex_iterator(np.shape(X))]).reshape(np.shape(X))
        subs_v = [(X[I],Y[I]) for I in multiindex_iterator(np.shape(X))]
        function_source_code = 'lambda Y:{0}'.format(repr(F.subs(subs_v)))
    # Function domain is 1-tensor or higher and function codomain is 1-tensor or higher
    else:
        Y = np.array([variable('Y[{0}]'.format(','.join(itertools.imap(str,I)))) for I in multiindex_iterator(np.shape(X))]).reshape(np.shape(X))
        subs_v = [(X[I],Y[I]) for I in multiindex_iterator(np.shape(X))]
        G = np.array([F[I].subs(subs_v) for I in multiindex_iterator(np.shape(F))]).reshape(np.shape(F))
        function_source_code = 'lambda Y:{0}'.format(repr(G))

    for from_string,to_string in replacement_d.iteritems():
        function_source_code = function_source_code.replace(from_string, to_string)
    if print_stuff:
        print 'function_source_code =', function_source_code

    compiled_function = eval(function_source_code)
    if print_stuff:
        print 'function constants:', compiled_function.func_code.co_consts
        print 'disassembled function code:\n', dis.dis(compiled_function.func_code.co_code)
    return compiled_function

if __name__ == '__main__':
    import sys

    x = variable('x')
    y = x**sympy.Rational(1,7)
    y_ = lambdify(y, x)
    for x_ in np.linspace(0.0, 10.0, 123):
        assert y_(x_) == x_**(1/7)
    print ''

    v = tensor('v', (3,))
    f = np.sum(np.square(v))
    f_ = lambdify(f, v)
    for v_ in itertools.product(*map(np.linspace, [-1.0]*len(v), [1.0]*len(v), [23]*len(v))):
        assert f_(v_) == sum(v_[i]**2 for i in xrange(len(v)))
    print ''

    phi = v / sympy.sqrt(np.sum(np.square(v)))
    phi_ = lambdify(phi, v)
    for v_ in itertools.product(*map(np.linspace, [-1.0]*len(v), [1.0]*len(v), [23]*len(v))):
        norm_v_ = np.linalg.norm(v_)
        # Avoid divide by zero or near zero.
        if norm_v_ < 1.0e-10:
            continue
        # assert all(phi_(v_) == np.array([v_[i] / norm_v_ for i in xrange(len(v))]))
        max_abs_error = np.max(np.abs(phi_(v_) - np.array([v_[i] / norm_v_ for i in xrange(len(v))])))
        assert max_abs_error == 0.0, 'v_ = {0}, max_abs_error = {1}'.format(v_, max_abs_error)
    print ''

    M = tensor('M', (2,3))
    A = M.T.dot(M)
    A_ = lambdify(A, M)
    for _ in xrange(1000):
        M_ = np.random.randn(2,3)
        max_abs_error = np.max(np.abs(A_(M_) - M_.T.dot(M_)))
        assert max_abs_error < 1.0e-14, 'M_ = {0}, max_abs_error = {1}'.format(M_, max_abs_error)
    print ''

    print 'passed all tests'

    sys.exit(0)

