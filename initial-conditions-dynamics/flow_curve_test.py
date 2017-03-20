import numpy as np
import scipy.integrate
import sympy as sp

# This is the hamiltonian (energy) function.
def hamiltonian (X, sqrt=np.sqrt):
    assert len(X) == 6, "X must be a 6-vector"
    x = X[0]
    y = X[1]
    z = X[2]
    p_x = X[3]
    p_y = X[4]
    p_z = X[5]
    alpha = 2.0/np.pi
    # alpha = 1.0
    r_squared = x**2 + y**2
    mu = r_squared**2 + 16.0*z**2
    P_x = p_x - 0.5*y*p_z
    P_y = p_y + 0.5*x*p_z
    return 0.5*(P_x**2 + P_y**2) - alpha/sqrt(mu)

# -\omega*dH is the hamiltonian vector field for this system
# t is the time at which to evaluate the flow.  This particular vector field is independent of time.
# X is the list of coordinates [x, y, z, p_x, p_y, p_z]
def hamiltonian_vector_field (t, X):
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
    # beta = 1.0/16.0
    beta = 16.0
    mu = r**2 + beta*z**2
    alpha = 2.0/np.pi
    # alpha = 1.0
    alpha_times_mu_to_neg_three_halves = alpha*mu**(-1.5)
    return np.array([P_x, \
                     P_y, \
                      0.5*x*P_y - 0.5*y*P_x, \
                     -0.5*P_y*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*x, \
                      0.5*P_x*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*y, \
                     -16.0*alpha_times_mu_to_neg_three_halves*z],
                     dtype=float)

def initial_condition ():
    # alpha = 1.0, beta = 1.0/16.0
    #for quasi-periodic use [1, 1, 4*3^(.5),0, 0] and t_0=273.5
    # (p_theta, r_0, z_0, p_r_0, p_z_0) = (1.0, 1.0, 4.0*3.0**0.5, 0.0, 0.0)
    # #this is for alpha=1 and theta=0
    # x_0 = r_0
    # y_0 = 0.0
    # z_0 = z_0
    # p_x_0 = p_r_0
    # p_y_0 = p_theta/r_0
    # p_z_0 = p_z_0
    # X_0 = np.array([x_0, y_0, z_0, p_x_0, p_y_0, p_z_0], dtype=float)
    # print("before perturbation: X_0 = {0}".format(X_0))
    # # X_0[4] += 5.0
    # # print("after perturbation:  X_0 = {0}".format(X_0))

    # X_0 = np.array([1.0, 0.0, 4.0*3.0**0.5, 0.0, 1.0, 0.0])

    # alpha = 2/pi, beta = 16

    # Symbolically solve H(1,0,0,0,1,p_z) = 0 for p_z.
    p_z = sp.var('p_z')
    zero = sp.Integer(0)
    one = sp.Integer(1)
    H = hamiltonian(np.array([one, zero, zero, zero, one, p_z], dtype=object), sqrt=sp.sqrt)
    print('H = {0}'.format(H))
    p_z_solution = np.max(sp.solve(H, p_z))
    print('p_z = {0}'.format(p_z_solution))
    X_0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, p_z_solution])

    return X_0

def compute_flow_curve (delta_t):
    period = 70
    # period = 273.5
    # sample_count = 2735
    t_0 = 0.0
    X_0 = initial_condition()
    print("Computing flow curve for time duration {0}, from initial condition {1}.".format(period, X_0))

    # Taken from http://stackoverflow.com/questions/16973036/odd-scipy-ode-integration-error
    ode = scipy.integrate.ode(hamiltonian_vector_field)
    ode.set_integrator('vode', nsteps=500, method='bdf') # This seems faster than dopri5
    # ode.set_integrator('dopri5', nsteps=500)
    ode.set_initial_value(X_0, t_0)

    t_v = [t_0]
    X_v_as_list = [X_0]
    while ode.successful() and ode.t < period:
        ode.integrate(ode.t + delta_t)
        # print(ode.t)
        t_v.append(ode.t)
        X_v_as_list.append(ode.y)

    return np.copy(X_v_as_list), t_v, X_v_as_list[-1], len(X_v_as_list), X_0

if __name__ == '__main__':
    X_v, t_v, period, sample_count, X_0 = compute_flow_curve(0.01)

    H_v = np.apply_along_axis(hamiltonian, 1, X_v)
    # half of squared distance from initial condition
    Q_v = 0.5 * np.sum(np.square(X_v - X_0), axis=-1)

    print(X_v)
    print(np.shape(X_v))
    print('H(0) = {0}'.format(H_v[0]))

    import matplotlib.pyplot as plt

    fig,axis_vv = plt.subplots(2, 2, squeeze=False, figsize=(20,20))

    axis = axis_vv[0][0]
    axis.set_title('(x,y) curve')
    axis.set_aspect('equal')
    axis.plot(X_v[:,0], X_v[:,1])

    axis = axis_vv[0][1]
    axis.set_title('(t,z) curve')
    axis.plot(t_v, X_v[:,2])

    axis = axis_vv[1][0]
    axis.set_title('(t,H) curve')
    axis.plot(t_v, H_v)
    axis.axhline(H_v[0])

    axis = axis_vv[1][1]
    axis.set_title('(t,Q) curve')
    axis.semilogy(t_v, Q_v)

    plt.savefig('flow_curve_test.png')

