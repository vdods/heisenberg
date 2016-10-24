import math

# This is the hamiltonian (energy) function.
def hamiltonian (X):
    assert len(X) == 6, "X must be a 6-vector"
    x = X[0]
    y = X[1]
    z = X[2]
    p_x = X[3]
    p_y = X[4]
    p_z = X[5]
    # alpha = 2.0/math.pi
    alpha = 1.0
    r_squared = x**2 + y**2
    mu = r_squared**2 + 16.0*z**2
    P_x = p_x - 0.5*y*p_z
    P_y = p_y + 0.5*x*p_z
    return 0.5*(P_x**2 + P_y**2) - alpha/math.sqrt(mu)

# -\omega*dH is the hamiltonian vector field for this system
# X is the list of coordinates [x, y, z, p_x, p_y, p_z]
# t is the time at which to evaluate the flow.  This particular vector field is independent of time.
def hamiltonian_vector_field (X, t):
    assert len(X) == 6, "must have 6 coordinates"
    x = X[0]
    y = X[1]
    z = X[2]
    p_x = X[3]
    p_y = X[4]
    p_z = X[5]
    P_x = p_x - 0.5*y*p_z
    P_y = p_y + 0.5*x*p_z
    r = x**2 + y**2
    beta = 1.0/16.0
    mu = r**2 + beta*z**2
    # alpha = 2.0/math.pi
    alpha = 1.0
    alpha_times_mu_to_neg_three_halves = alpha*mu**(-1.5)
    import numpy
    return numpy.array([P_x, \
                        P_y, \
                         0.5*x*P_y - 0.5*y*P_x, \
                        -0.5*P_y*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*x, \
                         0.5*P_x*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*y, \
                        -16.0*alpha_times_mu_to_neg_three_halves*z],
                        dtype=float)

def coreys_initial_condition ():
    #for quasi-periodic use [1, 1, 4*3^(.5),0, 0] and t_0=273.5
    (p_theta, r_0, z_0, p_r_0, p_z_0) = (1.0, 1.0, 4.0*3.0**0.5, 0.0, 0.0)
    #this is for alpha=1 and theta=0
    x_0 = r_0
    y_0 = 0.0
    z_0 = z_0
    p_x_0 = p_r_0
    p_y_0 = p_theta/r_0
    p_z_0 = p_z_0
    import numpy
    X_0 = numpy.array([x_0, y_0, z_0, p_x_0, p_y_0, p_z_0], dtype=float)
    # print "X_0 = {0}".format(X_0)
    return X_0

def compute_coreys_flow_curve ():
    period = 273.5
    sample_count = 2735
    X_0 = coreys_initial_condition()
    print "Computing flow curve for time duration {0}, with {1} samples, from initial condition {2}.".format(period, sample_count, X_0)
    import vector_field
    (Xs,Ts) = vector_field.compute_flow_curve(hamiltonian_vector_field, X_0, 0.0, period, sample_count)
    return Xs,Ts,period,sample_count

if __name__ == '__main__':
    X_v,t_v,period,sample_count = compute_coreys_flow_curve()

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(X_v[0,:], X_v[1,:])
    plt.show()

