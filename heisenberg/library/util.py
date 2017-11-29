import numpy as np

def polynomial_generator (generator, degree):
    """Returns np.array([1, generator, generator**2, ..., generator**degree])."""
    return np.array([generator**i for i in range(degree+1)])

def quadratic_min_time_parameterized (t_v, f_v):
    """
    Fits a quadratic exactly to the points (t_v[0],f_v[0]), (t_v[1],f_v[1]), (t_v[2],f_v[2])
    in the form of a numpy.ndarray with shape (3,), whose components are the coefficients for
    the quadratic.  In particular, if the quadratic is q(t) = q_0 + q_1*t + q_2*t**2, then
    the return value is

        np.array([q_0, q_1, q_2]).
    """
    assert len(t_v) == len(f_v), 't_v must have same length as f_v'
    assert all(t_i < t_j for t_i,t_j in zip(t_v[:-1],t_v[1:])), 't_v must be strictly increasing'
    assert len(f_v) == 3, 'require 3 values and the middle value must be a local min'
    assert f_v[0] > f_v[1] and f_v[1] < f_v[2]

    # Solve for the coefficients of the quadratic q(t) := q_v[0]*t^0 + q_v[1]*t^1 + q_v[2]*t^2
    # using the specified time values t_v and function values f_v;
    #     q(t_v[0]) = f_v[0]
    #     q(t_v[1]) = f_v[1]
    #     q(t_v[2]) = f_v[2]
    T = np.array([polynomial_generator(t,2) for t in t_v])
    q_v = np.linalg.solve(T, f_v)
    assert np.allclose(np.dot(T, q_v), f_v), 'q_v failed to solve intended equation T*q_v == f_v'
    # This is the critical point of q.
    t_min = -0.5*q_v[1]/q_v[2]
    assert t_v[0] < t_min < t_v[-1], 't_min should be bounded by first and last t_v values because f_v[1] is a local min.'
    # This is the corresponding value q(t_min)
    q_min = np.dot(polynomial_generator(t_min,2), q_v)

    return t_min,q_min

def exp_quadratic_min_time_parameterized (t_v, f_v):
    # The choice of epsilon should depend on the function being fit.
    epsilon = 1.0e-10
    t_min,q_min = quadratic_min_time_parameterized(t_v, np.log(f_v + epsilon))
    q_min = np.exp(q_min) - epsilon
    # Clip q_min at 0 just in case, since nonnegative is the whole point of this fanciness.
    if q_min < 0.0:
        q_min = 0.0
    return t_min,q_min

# Set numpy to print floats with full precision in scientific notation.
def float_formatter (x):
    return '{0:.17e}'.format(x)

def ndarray_as_single_line_string (A):
    """
    Normally a numpy.ndarray will be printed with spaces separating the elements
    and newlines separating the rows.  This function does the same but with commas
    separating elements and rows.  There should be no spaces in the returned string.
    """
    if len(A.shape) == 0:
        return float_formatter(A)
    else:
        return '[' + ','.join(ndarray_as_single_line_string(a) for a in A) + ']'

