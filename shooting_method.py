import abc
import itertools
import library.gradient_descent
import numpy as np
import scipy.integrate
import scipy.linalg

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
"""

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

    def hamiltonian (self, X):
        assert len(X) == 6, "X must be a 6-vector"
        x = X[0]
        y = X[1]
        z = X[2]
        p_x = X[3]
        p_y = X[4]
        p_z = X[5]
        # alpha = 2.0/np.pi
        alpha = 1.0
        r_squared = x**2 + y**2
        mu = r_squared**2 + 16.0*z**2
        P_x = p_x - 0.5*y*p_z
        P_y = p_y + 0.5*x*p_z
        return 0.5*(P_x**2 + P_y**2) - alpha/np.sqrt(mu)

    # \omega^-1 * dH (i.e. the symplectic gradient of H) is the hamiltonian vector field for this system.
    # X is the list of coordinates [x, y, z, p_x, p_y, p_z].
    # t is the time at which to evaluate the flow.  This particular vector field is independent of time.
    #
    # If the tautological one-form is
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
    def hamiltonian_vector_field (self, X, t):
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
        return np.array(
            [
                P_x,
                P_y,
                0.5*x*P_y - 0.5*y*P_x,
                -0.5*P_y*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*x,
                0.5*P_x*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*y,
                -16.0*alpha_times_mu_to_neg_three_halves*z
            ],
            dtype=float
        )

    @staticmethod
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
        X_0 = np.array([x_0, y_0, z_0, p_x_0, p_y_0, p_z_0], dtype=float)
        # print "X_0 = {0}".format(X_0)
        return X_0

class KeplerDynamicsContext(DynamicsContext):
    def __init__ (self):
        pass

    def configuration_space_dimension (self):
        return 2

    def hamiltonian (self, X, t):
        # Assume the planet has unit mass.
        q = X[0:2]
        p = X[2:4]
        # H = kinetic energy + potential energy
        return 0.5*np.sum(np.square(p)) - np.sum(np.square(q))**(-0.5)

    def hamiltonian_vector_field (self, X, t):
        q = X[0:2]
        p = X[2:4]
        factor = -0.5*np.sum(np.square(q))**(-1.5)
        return np.array(
            [
                p[0],
                p[1],
                factor*q[0],
                factor*q[1]
            ],
            dtype=float
        )

class NBodyDynamicsContext(DynamicsContext):
    def __init__ (self, *, mass_v, gravitational_constant):
        assert len(mass_v.shape) == 1
        assert mass_v.shape[0] > 0
        assert all(mass > 0.0 for mass in mass_v)
        assert gravitational_constant > 0.0

        self.__mass_v = mass_v
        self.__gravitational_constant = gravitational_constant

        self.__n = mass_v.shape[0]
        self.__inverse_mass_v = 1.0 / mass_v
        self.__body_index_pair_v = [(i,j) for i,j in itertools.product(range(self.__n), range(self.__n)) if i < j]

    def configuration_space_dimension (self):
        return 2*self.__n

    def hamiltonian (self, X, t):
        q_v = X[:self.__n].reshape(self.__n, 2) # positions
        p_v = X[self.__n:].reshape(self.__n, 2) # momenta
        # Kinetic energy
        K = 0.5*np.dot(np.sum(np.square(p_v), axis=-1), self.__inverse_mass_v)
        # Potential energy
        V = -self.__gravitational_constant*sum(
            self.__mass_v[i]*self.__mass_v[j]*np.sum(np.square(q_v[i]-q_v[v]))**(-0.5)
            for i,j in self.__body_index_pair_v
        )
        # Total energy
        return K + V

    def hamiltonian_vector_field (self, X, t):
        # q = X[0:2]
        # p = X[2:4]
        # factor = -0.5*np.sum(np.square(q))**(-1.5) #*(1.0+self.sin_amplitude*np.sin(t*self.omega))
        # return np.array(
        #     [
        #         p[0],
        #         p[1],
        #         factor*q[0],
        #         factor*q[1]
        #     ],
        #     dtype=float
        # )
        csd = self.phase_space_dimension()

        q_v = X[:csd].reshape(self.__n, 2) # positions
        p_v = X[csd:].reshape(self.__n, 2) # momenta
        retval = np.ndarray((self.phase_space_dimension(),), dtype=float)

        for i in range(self.__n):
            retval[2*i:2*(i+1)]         = p_v[i]*self.__inverse_mass_v[i]
            retval[csd+2*i:csd+2*(i+1)] = sum(
                self.__mass_v[i]*self.__mass_v[j]*np.sum(np.square(q_v[i]-q_v[j]))**(-1.5)
                for i,j in self.__body_index_pair_v
            )
        retval[csd:] *= -0.5*self.__gravitational_constant

        return retval

class ShootingMethodObjective:
    def __init__ (self, *, dynamics_context, X_0, t_v):
        self.__dynamics_context     = dynamics_context
        self.X_0                    = X_0
        self.__X_v                  = None
        self.t_v                    = t_v
        self.__s_v                  = None
        self.__s_global_min_index   = None
        self.__objective            = None

    def configuration_space_dimension (self):
        return self.__dynamics_context.configuration_space_dimension()

    def flow_curve (self):
        if self.__X_v is None:
            # Compute the flow curve using X_0 as initial condition
            self.__X_v = scipy.integrate.odeint(lambda X,t:self.__dynamics_context.hamiltonian_vector_field(X,t), self.X_0, self.t_v, full_output=False)
            # print('infodict:')
            # print(infodict)
        return self.__X_v

    def squared_distance_function (self):
        if self.__s_v is None:
            X_0 = self.X_0
            X_v = self.flow_curve()
            # Let s denote squared distance function s(t) := 1/2 |X_0 - flow_of_X_0(t))|^2
            self.__s_v = 0.5 * np.sum(np.square(X_v - X_0), axis=-1)
        return self.__s_v

    def objective (self):
        if self.__objective is None:
            self.compute_s_global_min_index_and_objective()
        return self.__objective

    def s_global_min_index (self):
        if self.__s_global_min_index is None:
            self.compute_s_global_min_index_and_objective()
        return self.__s_global_min_index

    # def compute_f_and_R (t_v, X_v):
    def __call__ (self):
        # if self.__objective is None:
        #     # X_0 = self.X_0
        #     # X_v = self.flow_curve()
        #     # s_v = self.squared_distance_function()

        #     # local_min_index_v   = [i for i in range(1,len(s_v)-1) if s_v[i-1] > s_v[i] and s_v[i] < s_v[i+1]]
        #     # s_local_min_v       = [s_v[i] for i in local_min_index_v]
        #     # try:
        #     #     s_local_min_min_index   = np.argmin(s_local_min_v)
        #     #     self.s_global_min_index = s_global_min_index = local_min_index_v[s_local_min_min_index]
        #     #     self.__objective        = s_v[s_global_min_index]
        #     # except ValueError:
        #     #     self.s_global_min_index = None
        #     #     self.__objective        = np.inf
        #     self.compute_s_global_min_index_and_objective()
        # return self.__objective#,R
        return self.objective()

    def compute_s_global_min_index_and_objective (self):
        X_0 = self.X_0
        X_v = self.flow_curve()
        s_v = self.squared_distance_function()

        local_min_index_v   = [i for i in range(1,len(s_v)-1) if s_v[i-1] > s_v[i] and s_v[i] < s_v[i+1]]
        s_local_min_v       = [s_v[i] for i in local_min_index_v]
        try:
            s_local_min_min_index       = np.argmin(s_local_min_v)
            self.__s_global_min_index   = _s_global_min_index = local_min_index_v[s_local_min_min_index]
        except ValueError:
            # self.__s_global_min_index   = None
            # self.__objective            = np.inf
            # If there was no local min, then use the last time value
            self.__s_global_min_index = len(s_v)-1
        self.__objective            = s_v[self.__s_global_min_index]            

def evaluate_shooting_method_objective (dynamics_context, X_0, t_v):
    return ShootingMethodObjective(dynamics_context=dynamics_context, X_0=X_0, t_v=t_v)()

def D (*, dynamics_context, X_0, t_v, epsilon, rng):
    f_0 = evaluate_shooting_method_objective(dynamics_context, X_0, t_v)

    # # grad R is a 6x6 matrix, so there are 6**2 unknowns, but bump that up by a factor of 10.
    # sample_count = 10*dynamics_context.phase_space_dimension()**2

    # 5 samples per unknown.
    sample_count = 5*dynamics_context.phase_space_dimension()

    # R_v = []

    perturbation_v = np.ndarray((sample_count,X_0.shape[0]), dtype=float)
    f_v = np.ndarray((sample_count,), dtype=float)
    i = 0
    while i < sample_count: 
        perturbation = rng.randn(X_0.shape[0])
        # f,R = compute_f_and_R(t_v, X_0_perturbed)
        f = evaluate_shooting_method_objective(dynamics_context, X_0+perturbation, t_v)
        if np.isfinite(f):
            perturbation_v[i]   = perturbation
            f_v[i]              = f
            i                  += 1
        # R_v.append(R)

    # Taylor's theorem states that
    #   f(X) = f(X_0) + Df(X_0)*(X-X_0) + o(X-X_0),
    # which is equivalent to
    #   Df(X_0)*(X-X_0) = f(X)-f(X_0) + o(X-X_0).

    Df = scipy.linalg.lstsq(perturbation_v, f_v-f_0)[0]
    # DR = scipy.linalg.lstsq(perturbation_v, R_v-R_0)[0]

    # return Df,DR
    return Df

def make_time_array (*, t_max, t_count):
    return np.linspace(0.0, t_max, num=t_count)

def draw_arrow (axis, basepoint, direction, *args, **kwargs):
    direction_norm = np.linalg.norm(direction)
    axis.arrow(basepoint[0], basepoint[1], 0.9*direction[0], 0.9*direction[1], head_width=0.05*direction_norm, head_length=0.1*direction_norm, *args, **kwargs)
    arrowhead_point = basepoint + direction
    return arrowhead_point

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # X_0 = HeisenbergDynamicsContext.coreys_initial_condition()


    dynamics_context = KeplerDynamicsContext()
    # X_0 = np.array([1.0, 0.0, 0.0, 0.5]) # closed
    X_0 = np.array([1.0, 0.0, 0.0, 2.0]) # open
    t_v = make_time_array(t_max=6.0, t_count=4000)
    optimization_epsilon = 0.001
    position_slice_v = [slice(0,2)]
    momentum_slice_v = [slice(2,4)]


    # dynamics_context = NBodyDynamicsContext([1.0, 1.0])
    # X_0 = np.array([
    #      1.0,  0.0,
    #     -1.0,  0.0,
    #      0.0,  1.0,
    #      0.0, -1.0,
    # ])
    # t_v = make_time_array(t_max=10.0, t_count=4000)
    # optimization_epsilon = 0.001

    row_count = 1
    col_count = 3
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(5*col_count,5*row_count))

    def plot_stuff (*, axis, X_0, t_v, position_slice_v, momentum_slice_v):
        assert len(position_slice_v) == len(momentum_slice_v)

        print('plot_stuff(); X_0 = {0}'.format(X_0))

        shooting_method_objective = ShootingMethodObjective(dynamics_context=dynamics_context, X_0=X_0, t_v=t_v)
        Dobj = D(dynamics_context=dynamics_context, X_0=X_0, t_v=t_v, epsilon=0.0001, rng=np.random.RandomState(42))
        X_v = shooting_method_objective.flow_curve()

        for i,(position_slice,momentum_slice) in enumerate(zip(position_slice_v,momentum_slice_v)):
            position_v = X_v[:,position_slice]
            # axis.plot(X_v[:,0], X_v[:,1], color='black')
            axis.plot(position_v[:,0], position_v[:,1], color='black')

            arrowhead_point = draw_arrow(axis, X_0[position_slice], X_0[momentum_slice], color='green')
            draw_arrow(axis, X_0[position_slice], Dobj[position_slice], color='blue')
            draw_arrow(axis, arrowhead_point, Dobj[momentum_slice], color='blue')

        print('shooting_method_objective.s_global_min_index() = {0}'.format(shooting_method_objective.s_global_min_index()))
        X_closest = X_v[shooting_method_objective.s_global_min_index()]
        print('X_closest = {0}'.format(X_closest))

        for i,(position_slice,momentum_slice) in enumerate(zip(position_slice_v,momentum_slice_v)):
            draw_arrow(axis, X_closest[position_slice], X_closest[momentum_slice], color='red')

        axis.set_title('phase space dist: {0:.2e}'.format(np.linalg.norm(X_0 - X_closest)))

        # draw_arrow(axis, X_v[-1,0:2], X_v[-1,2:4], color='blue')

        # axis.set_xlim(-1.5e0, 1.8e0)
        # axis.set_ylim(-1.2e0, 1.4e0)
        axis.set_aspect(1)

    plot_stuff(axis=axis_vv[0][0], X_0=X_0, t_v=t_v, position_slice_v=position_slice_v, momentum_slice_v=position_slice_v)

    def obj (X):
        return ShootingMethodObjective(dynamics_context=dynamics_context, X_0=X, t_v=t_v)()

    optimizer = library.gradient_descent.BlindGradientDescent(
        obj,
        lambda X_0:D(dynamics_context=dynamics_context, X_0=X_0, t_v=t_v, epsilon=0.0001, rng=np.random.RandomState(42)),
        X_0,
        optimization_epsilon
    )
    X_opt = X_0
    optimization_iteration_count = 100
    print('running optimizer for {0} iterations'.format(optimization_iteration_count))
    try:
        for optimization_iteration_index in range(optimization_iteration_count):
            X_opt = optimizer.compute_next_step(X_opt, True)
            print('result of iteration {0}: objective function value: {1}'.format(optimization_iteration_index, optimizer.obj_history_v[-1]))
    except KeyboardInterrupt:
        print('got KeyboardInterrupt -- halting optimization')

    # Take the best X_opt
    X_opt = optimizer.parameter_history_v[np.argmin(optimizer.obj_history_v)]

    plot_stuff(axis=axis_vv[0][1], X_0=X_opt, t_v=t_v, position_slice_v=position_slice_v, momentum_slice_v=momentum_slice_v)

    if True:
        axis = axis_vv[0][2]
        axis.set_title('objective function history')
        axis.semilogy(optimizer.obj_history_v)
        axis.axvline(np.argmin(optimizer.obj_history_v), color='green')

    fig.tight_layout()
    plt.savefig('shooting_method.png')
