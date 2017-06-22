import abc
import itertools
import library.monte_carlo
import numpy as np
import scipy.integrate
import scipy.linalg
import sympy as sp
import time
import vorpy.symbolic

"""
Notes

Define "return map" R : T^* Q -> T^* Q (really R^3xR^3 -> R^3xR^3, because it's coordinate dependent):
R(q,p) is defined as the closest point (in the coordinate chart R^3xR^3 for T^* Q) to (q,p) in the
sequence of points in the solution to the orbital curve for initial condition (q,p).

Define f : T^* Q -> R, (q,p) |-> 1/2 * |(q,p) - R(q,p)|^2

Use gradient descent to find critical points of f.

The gradient of f depends on the gradient of R.  This can be computed numerically using a least-squares
approximation of the first-order Taylor polynomial of R.

Select initial conditions for the gradient descent to be on the H(q,p) = 0 submanifold, probably
by picking 5 coordinates at random and solving for the 6th.

Symmetry condition:  Define symmetry via map Omega : T^* Q -> T^* Q (e.g. rotation through 2*pi/3).
Define R_Omega to give point closest to Omega(q,p).  Then f_Omega is defined as

    f_Omega(q,p) := 1/2 * |Omega(q,p) - R_Omega(q,p)|^2,

and the gradient of f_Omega depends on the gradient of Omega and R_Omega.

TODO
-   Use energy-conserving integrator
-   The 7 fold solution is super close to closing, and the optimization doesn't improve much.
    Perturb it (but keep it zero-energy) and see if the optimizer can close it back up.
-   I think the period detection isn't fully correct for the following reason.  Often times
    a curve will be quasi-periodic, or have a really high order of symmetry resulting in
    a very high period.  Probably what we actually want to happen is that the first reasonable
    candidate for period is selected, so that the symmetry order is relatively low, and the
    optimizer then tries to close up that curve.

    Also, we must guarantee that the period computation picks analogous points on the curve,
    meaning that they come from similar time values (and not e.g. several loops later in time).
"""

def define_canonical_symplectic_form_and_inverse (*, configuration_space_dimension, dtype):
    # If the tautological one-form on the cotangent bundle is
    #   tau := p dq
    # then the symplectic form is
    #   omega := -dtau = -dq wedge dp
    # which, in the coordinates (q_0, q_1, p_0, p_1), has the matrix
    #   [  0  0 -1  0 ]
    #   [  0  0  0 -1 ]
    #   [  1  0  0  0 ]
    #   [  0  1  0  0 ],
    # or in matrix notation, with I denoting the 2x2 identity matrix,
    #   [  0 -I ]
    #   [  I  0 ],

    assert configuration_space_dimension > 0

    # Abbreviations
    csd = configuration_space_dimension
    psd = 2*csd

    canonical_symplectic_form = np.ndarray((psd,psd), dtype=dtype)

    # Fill the whole thing with zeros.
    canonical_symplectic_form.fill(dtype(0))

    # Upper right block diagonal is -1, lower left block diagonal is 1.
    for i in range(csd):
        canonical_symplectic_form[i,csd+i] = dtype(-1)
        canonical_symplectic_form[csd+i,i] = dtype( 1)

    canonical_symplectic_form_inverse = -canonical_symplectic_form

    return canonical_symplectic_form,canonical_symplectic_form_inverse

def symplectic_gradient_of (F, X, *, canonical_symplectic_form_inverse=None, dtype=None):
    assert len(X)%2 == 0, 'X must be a phase space element, which in particular means it must be even dimensional.'

    if canonical_symplectic_form_inverse is None:
        assert dtype is not None, 'If canonical_symplectic_form_inverse is None, then dtype must not be None.'
        _,canonical_symplectic_form_inverse = define_canonical_symplectic_form_and_inverse(configuration_space_dimension=X.shape[0]//2, dtype=dtype)

    return np.dot(canonical_symplectic_form_inverse, vorpy.symbolic.D(F,X))

def quadratic_min (f_v):
    assert len(f_v) == 3, 'require 3 values'
    c = f_v[1]
    b = 0.5*(f_v[2] - f_v[0])
    a = 0.5*(f_v[2] + f_v[0]) - f_v[1]
    x = -0.5*b/a
    return a*x**2 + b*x + c

class DynamicsContext(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def configuration_space_dimension (self):
        pass

    @abc.abstractmethod
    def hamiltonian (self, X):
        pass

    @abc.abstractmethod
    def hamiltonian_vector_field (self, X, t):
        pass

    def phase_space_dimension (self):
        return 2*self.configuration_space_dimension()

class HeisenbergDynamicsContext(DynamicsContext):
    def __init__ (self):
        pass

    def configuration_space_dimension (self):
        return 3

    # This is the hamiltonian (energy) function.
    @staticmethod
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

    # \omega^-1 * dH (i.e. the symplectic gradient of H) is the hamiltonian vector field for this system.
    # X is the list of coordinates [x, y, z, p_x, p_y, p_z].
    # t is the time at which to evaluate the flow.  This particular vector field is independent of time.
    #
    # If the tautological one-form on the cotangent bundle is
    #   tau := p dq
    # then the symplectic form is
    #   omega := -dtau = -dq wedge dp
    # which, in the coordinates (q_0, q_1, p_0, p_1), has the matrix
    #   [  0  0 -1  0 ]
    #   [  0  0  0 -1 ]
    #   [  1  0  0  0 ]
    #   [  0  1  0  0 ],
    # or in matrix notation, with I denoting the 2x2 identity matrix,
    #   [  0 -I ]
    #   [  I  0 ],
    # having inverse
    #   [  0  I ]
    #   [ -I  0 ].
    # With dH:
    #   dH = dH/dq * dq + dH/dp * dp,    (here, dH/dq denotes the partial of H w.r.t. q)
    # or expressed in coordinates as
    #   [ dH/dq ]
    #   [ dH/dp ]
    # it follows that the sympletic gradient of H is
    #   dH/dp * dq - dH/dq * dp
    # or expressed in coordinates as
    #   [  dH/dp ]
    #   [ -dH/dq ],
    # which is Hamilton's equations.
    def hamiltonian_vector_field (self, t, X): # NOTE: t comes first, because of the convention of scipy.integrate.ode
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
        return np.array(
            [
                P_x, \
                P_y, \
                 0.5*x*P_y - 0.5*y*P_x, \
                -0.5*P_y*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*x, \
                 0.5*P_x*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*y, \
                -beta*alpha_times_mu_to_neg_three_halves*z
            ],
            dtype=float
        )

    @staticmethod
    def initial_condition ():
        # alpha = 2/pi, beta = 16

        # Symbolically solve H(1,0,0,0,1,p_z) = 0 for p_z.
        p_z = sp.var('p_z')
        zero = sp.Integer(0)
        one = sp.Integer(1)
        # H = HeisenbergDynamicsContext.hamiltonian(np.array([one, zero, zero, zero, one, p_z], dtype=object), sqrt=sp.sqrt, pi=sp.pi)
        H = HeisenbergDynamicsContext.hamiltonian(np.array([one/2, zero, zero, zero, one, p_z], dtype=object), sqrt=sp.sqrt, pi=sp.pi)
        print('H = {0}'.format(H))
        p_z_solution = np.max(sp.solve(H, p_z))
        print('p_z = {0}'.format(p_z_solution))
        p_z_solution = float(p_z_solution)
        # X_0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, p_z_solution])
        X_0 = np.array([0.5, 0.0, 0.0, 0.0, 1.0, p_z_solution])

        return X_0

class ShootingMethodObjective:
    def __init__ (self, *, dynamics_context, X_0, t_max, t_delta):
        self.__dynamics_context     = dynamics_context
        self.X_0                    = X_0
        self.__X_v                  = None
        self.t_max                  = t_max
        self.t_delta                = t_delta
        self.__Q_v                  = None
        self.__Q_global_min_index   = None
        self.__objective            = None

    def configuration_space_dimension (self):
        return self.__dynamics_context.configuration_space_dimension()

    def flow_curve (self):
        if self.__X_v is None:
            # Compute the flow curve using X_0 as initial condition

            # Taken from http://stackoverflow.com/questions/16973036/odd-scipy-ode-integration-error
            ode = scipy.integrate.ode(self.__dynamics_context.hamiltonian_vector_field)
            # ode.set_integrator('vode', nsteps=500, method='bdf') # This seems faster than dopri5
            # ode.set_integrator('vode', nsteps=1000, method='bdf') # This seems faster than dopri5
            ode.set_integrator('dopri5', nsteps=500)
            ode.set_initial_value(self.X_0, 0.0)

            start_time = time.time()

            t_v = [0.0]
            X_v_as_list = [self.X_0]
            while ode.successful() and ode.t < t_max:
                ode.integrate(ode.t + t_delta)
                # print(ode.t)
                t_v.append(ode.t)
                X_v_as_list.append(ode.y)

            print('integration took {0} seconds'.format(time.time() - start_time))

            self.__t_v = t_v
            self.__X_v = np.copy(X_v_as_list)
        return self.__X_v

    def t_v (self):
        if self.__t_v is None:
            self.flow_curve()
            assert self.__t_v is not None
        return self.__t_v

    def squared_distance_function (self):
        if self.__Q_v is None:
            X_0 = self.X_0
            X_v = self.flow_curve()
            # Let s denote squared distance function s(t) := 1/2 |X_0 - flow_of_X_0(t))|^2
            self.__Q_v = 0.5 * np.sum(np.square(X_v - X_0), axis=-1)
        return self.__Q_v

    def objective (self):
        if self.__objective is None:
            self.compute_Q_global_min_index_and_objective()
        return self.__objective

    def Q_global_min_index (self):
        if self.__Q_global_min_index is None:
            self.compute_Q_global_min_index_and_objective()
        return self.__Q_global_min_index

    def closest_approach_point (self):
        return self.flow_curve()[self.Q_global_min_index()]

    def __call__ (self):
        return self.objective()

    def compute_Q_global_min_index_and_objective (self):
        X_0                             = self.X_0
        X_v                             = self.flow_curve()
        self.__Q_v = Q_v                = self.squared_distance_function()

        local_min_index_v               = [i for i in range(1,len(Q_v)-1) if Q_v[i-1] > Q_v[i] and Q_v[i] < Q_v[i+1]]
        Q_local_min_v                   = [Q_v[i] for i in local_min_index_v]
        try:
            Q_local_min_min_index       = np.argmin(Q_local_min_v)
            self.__Q_global_min_index   = _Q_global_min_index = local_min_index_v[Q_local_min_min_index]
            if False:
                assert 1 <= _Q_global_min_index < len(Q_v)-1
                self.__objective        = quadratic_min(Q_v[_Q_global_min_index-1:_Q_global_min_index+2])
                # Some tests show this discrepancy to be on the order of 1.0e-9
                print('self.__objective - Q_v[_Q_global_min_index] = {0}'.format(self.__objective - Q_v[_Q_global_min_index]))
            else:
                self.__objective        = Q_v[_Q_global_min_index]
        except ValueError:
            # If there was no local min, then use the last time value
            self.__Q_global_min_index   = len(Q_v)-1
            self.__objective            = Q_v[self.__Q_global_min_index]

def evaluate_shooting_method_objective (dynamics_context, X_0, t_max, t_delta):
    return ShootingMethodObjective(dynamics_context=dynamics_context, X_0=X_0, t_max=t_max, t_delta=t_delta)()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dynamics_context = HeisenbergDynamicsContext()
    X_0 = HeisenbergDynamicsContext.initial_condition()
    t_max = 60.0
    t_delta = 0.01
    smo_0 = ShootingMethodObjective(dynamics_context=dynamics_context, X_0=X_0, t_max=t_max, t_delta=t_delta)
    flow_curve_0 = smo_0.flow_curve()

    optimizer = library.monte_carlo.MonteCarlo(lambda x_0:evaluate_shooting_method_objective(dynamics_context, x_0, t_max, t_delta), X_0, 1.0e-12, 1.0e-5, 12345)
    try:
        # for i in range(10000):
        for i in range(100):
            optimizer.compute_next_step()
            print('i = {0}, obj = {1}'.format(i, optimizer.obj_history_v[-1]))
    except KeyboardInterrupt:
        print('got KeyboardInterrupt -- halting optimization')        

    X_opt = optimizer.parameter_history_v[-1]
    smo_opt = ShootingMethodObjective(dynamics_context=dynamics_context, X_0=X_opt, t_max=t_max, t_delta=t_delta)
    flow_curve_opt = smo_opt.flow_curve()

    print('X_0 = {0}'.format(X_0))
    print('X_opt = {0}'.format(X_opt))
    print('flow_curve_0[0] = {0}'.format(flow_curve_0[0]))
    print('flow_curve_0[-1] = {0}'.format(flow_curve_0[-1]))
    print('flow_curve_opt[0] = {0}'.format(flow_curve_opt[0]))
    print('flow_curve_opt[-1] = {0}'.format(flow_curve_opt[-1]))

    def plot_stuff (*, axis_v, smo, name):
        flow_curve = smo.flow_curve()

        axis = axis_v[0]
        axis.set_title('{0} curve'.format(name))
        axis.plot(flow_curve[:,0], flow_curve[:,1])
        axis.plot(flow_curve[0,0], flow_curve[0,1], 'o', color='green', alpha=0.5)
        axis.plot(flow_curve[smo.Q_global_min_index(),0], flow_curve[smo.Q_global_min_index(),1], 'o', color='red', alpha=0.5)
        axis.set_aspect('equal')

        axis = axis_v[1]
        axis.set_title('squared distance')
        axis.semilogy(smo.t_v(), smo.squared_distance_function())
        axis.axvline(smo.t_v()[smo.Q_global_min_index()], color='green')

        axis = axis_v[2]
        axis.set_title('curve energy')
        axis.plot(smo.t_v(), np.apply_along_axis(HeisenbergDynamicsContext.hamiltonian, 1, flow_curve))

    row_count = 2
    col_count = 4
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(15*col_count,15*row_count))

    # axis = axis_vv[0][0]
    # axis.set_title('initial curve')
    # axis.plot(flow_curve_0[:,0], flow_curve_0[:,1])
    # axis.set_aspect('equal')

    plot_stuff(axis_v=axis_vv[0], smo=smo_0, name='initial')
    plot_stuff(axis_v=axis_vv[1], smo=smo_opt, name='optimized')

    axis = axis_vv[0][3]
    axis.set_title('objective function history')
    axis.semilogy(optimizer.obj_history_v)

    # axis = axis_vv[1][0]
    # axis.set_title('optimized curve')
    # axis.plot(flow_curve_opt[:,0], flow_curve_opt[:,1])
    # axis.set_aspect('equal')

    # axis = axis_vv[1][2]
    # axis.set_title('energy of optimized curve')
    # axis.plot(smo_opt.t_v(), np.apply_along_axis(HeisenbergDynamicsContext.hamiltonian, 1, flow_curve_opt))

    fig.tight_layout()
    filename = 'shooting_method_2.png'
    plt.savefig(filename)
    print('wrote to file "{0}"'.format(filename))
