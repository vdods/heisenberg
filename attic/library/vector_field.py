def compute_flow_curve (vector_field, X_0, T_0, duration, sample_count):
    """
    Computes a numerical approximation of a flow curve for the given vector field
    vector_field(X,T), using the given initial conditions (X_0,T_0), representing
    the curve, defined for time interval [0,duration], in the discrete form
        Xs = (X(t_0),...,X(t_{n-1})), 
        Ts = (t_0,...,t_{n-1}),
    where the time samples Ts are uniformly sampled, endpoint-inclusive, over the
    interval [0,duration].  The return value is (Xs,Ts).

    This function uses scipy.integrate.odeint, so see documentation for that function
    for more info regarding the vector_field, X_0, and T_0 parameters.
    """
    assert duration > 0, "duration must be positive"
    import numpy
    # Uniformly sample the interval [0,duration] for the time values Ts.
    Ts = numpy.linspace(T_0, T_0+duration, num=sample_count)
    import scipy.integrate
    Xs = scipy.integrate.odeint(vector_field, X_0, Ts)
    return (Xs,Ts)

