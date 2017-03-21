import numpy as np
import scipy.integrate
import sympy as sp
import time

# This is the hamiltonian (energy) function.
def hamiltonian (X, sqrt=np.sqrt, pi=np.pi):
    assert len(X) == 6, "X must be a 6-vector"
    x = X[0]
    y = X[1]
    z = X[2]
    p_x = X[3]
    p_y = X[4]
    p_z = X[5]
    alpha = 2/pi
    # alpha = 1.0
    beta = 16
    r_squared = x**2 + y**2
    mu = r_squared**2 + beta*z**2
    P_x = p_x - y*p_z/2
    P_y = p_y + x*p_z/2
    return (P_x**2 + P_y**2)/2 - alpha/sqrt(mu)

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
                     -beta*alpha_times_mu_to_neg_three_halves*z],
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
    H = hamiltonian(np.array([one, zero, zero, zero, one, p_z], dtype=object), sqrt=sp.sqrt, pi=sp.pi)
    print('H = {0}'.format(H))
    p_z_solution = np.max(sp.solve(H, p_z))
    print('p_z = {0}'.format(p_z_solution))
    p_z_solution = float(p_z_solution)
    X_0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, p_z_solution])

    return X_0

def compute_stuff (*, max_t, delta_t):
    max_t = 70
    # max_t = 273.5
    # sample_count = 2735
    t_0 = 0.0
    X_0 = initial_condition()
    print("Computing flow curve for time duration {0}, from initial condition {1}.".format(max_t, X_0))

    # Taken from http://stackoverflow.com/questions/16973036/odd-scipy-ode-integration-error
    ode = scipy.integrate.ode(hamiltonian_vector_field)
    ode.set_integrator('vode', nsteps=500, method='bdf') # This seems faster than dopri5
    # ode.set_integrator('dopri5', nsteps=500)
    ode.set_initial_value(X_0, t_0)

    start_time = time.time()

    t_v = [t_0]
    X_v_as_list = [X_0]
    while ode.successful() and ode.t < max_t:
        ode.integrate(ode.t + delta_t)
        # print(ode.t)
        t_v.append(ode.t)
        X_v_as_list.append(ode.y)

    print('integration took {0} seconds'.format(time.time() - start_time))

    X_v = np.copy(X_v_as_list)
    H_v = np.apply_along_axis(hamiltonian, 1, X_v)
    # half of squared distance from initial condition
    Q_v = 0.5 * np.sum(np.square(X_v - X_0), axis=-1)

    local_min_index_v   = [i for i in range(1,len(Q_v)-1) if Q_v[i-1] > Q_v[i] and Q_v[i] < Q_v[i+1]]
    Q_local_min_v       = [Q_v[i] for i in local_min_index_v]
    try:
        Q_local_min_min_index = np.argmin(Q_local_min_v)
        Q_global_min_index    = local_min_index_v[Q_local_min_min_index]
    except ValueError:
        # If there was no local min, then use the last time value
        Q_global_min_index    = len(Q_v)-1
    Q_min               = Q_v[Q_global_min_index]

    return t_v, X_v, H_v, Q_v, Q_global_min_index

if __name__ == '__main__':
    max_t = 70.0
    delta_t = 0.01

    t_v, X_v, H_v, Q_v, Q_global_min_index = compute_stuff(max_t=max_t, delta_t=delta_t)

    period = t_v[Q_global_min_index]
    Q_min = Q_v[Q_global_min_index]

    print(X_v)
    print(np.shape(X_v))
    print('H(0) = {0}'.format(H_v[0]))
    print('period = {0}'.format(period))
    print('Q_min = {0}'.format(Q_min))

    import matplotlib.pyplot as plt

    fig,axis_vv = plt.subplots(2, 2, squeeze=False, figsize=(20,20))

    axis = axis_vv[0][0]
    axis.set_title('(x,y) curve; period = {0}'.format(period))
    axis.set_aspect('equal')
    axis.axhline(0, color='black')
    axis.axvline(0, color='black')
    axis.plot(X_v[:,0], X_v[:,1])
    axis.plot(X_v[Q_global_min_index,0], X_v[Q_global_min_index,1], 'o', color='green')

    axis = axis_vv[0][1]
    axis.set_title('(t,z) curve')
    axis.axvline(period, color='green')
    axis.plot(t_v, X_v[:,2])

    axis = axis_vv[1][0]
    axis.set_title('(t,H) curve')
    axis.axvline(period, color='green')
    axis.axhline(H_v[0], color='black')
    axis.plot(t_v, H_v)

    axis = axis_vv[1][1]
    axis.set_title('(t,Q) curve; min = {0}'.format(Q_min))
    axis.axvline(period, color='green')
    axis.semilogy(t_v, Q_v)

    plt.savefig('flow_curve_test.png')

