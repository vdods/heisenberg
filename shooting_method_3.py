import abc
import itertools
import library.monte_carlo
import numpy as np
import sympy as sp
import time
import vorpy
import vorpy.symbolic
import vorpy.symplectic_integration

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
-   Examine following hypothesis: Z curve for closed orbit is a sine wave.
-   Examine following hypothesis: Z curve for quasi-periodic orbit is a sine wave of decreasing
    amplitude and increasing frequency, perhaps this can be expressed simply as being a rotated
    and dilated (and time-reparameterized) version of itself.
-   Replace 'X[0]', 'X[1]' names in solution for H = 0 with actual names of coordinates, specifically
    x,y,z,p_x,p_y,p_z.
-   If abs(H) escapes a certain threshold while integrating, either abort or decreate the timestep.
-   k-fold symmetry -- make the closest-approach-map measure from a 2pi/k-rotated phase space point
    in order to more efficiently find k-fold symmetric curves.
-   try to come up with uniform distribution on bounded portion of 2D parameter embedding space
    for search routine
-   Decide dt based on the abs(H) bound -- if it's above say 1.0e-4, then reduce dt.
-   When plotting closed curves, should probably only plot up to the t_min value, not the overlapping
    part.
-   Idea for defining a more-robust objective function: Compute t_min as before, but then define the
    objective function as the L_2 norm squared of the difference between overlapping segments.
    In more detail, the candidate curve extends past t_min, say until t_min+T for some positive T.
    Then define the objective function to be

        L_2(curve[0:T] - curve[t_min:t_min+T])^2

    so that curves that are nearly coincident for a positive time duration have a lower objective
    function than curves that merely have a single point that comes close to the initial condition.
-   Program to search for orbits in the 2D embedding space:
    -   Generate random points in the domain

            -sqrt(pi/4) <= p_x <= sqrt(pi/4)
                     -C <= p_y <= C

        for some arbitrary positive constant C, say 2.  Due to a discrete symmetry in the system
        (reflection), p_y can be taken to be nonnegative.  Thus the domain can be

            -sqrt(pi/4) <= p_x <= sqrt(pi/4)
                      0 <= p_y <= C

    -   For each of these, compute the embedding qp_0 into phase space and use that as the initial
        condition for the flow curve.  Use some fixed dt and max_time.
    -   For each flow curve, compute the following values:
        -   The objective function value (which could be NaN if the curve didn't go back toward itself)
        -   The t_min value (which for a closed curve would be its period)
        -   The upper bound on abs(H)
        -   The upper bound on abs(J - J_0)
        -   If it is [close to being] closed, then the order of its radial symmetry
        -   The deviation of its z(t) function from a sine wave (for closed, this always appears to
            be a sine wave, but this hasn't been checked).
    -   Visualize these two functions to determine if they are continuous and/or have other structure
        to them.  Perhaps there are regions which are continuous surrounding zeros the objective function.
"""

def polynomial_basis_vector (generator, degree):
    return np.array([generator**i for i in range(degree+1)])

def quadratic_min_time_parameterized (t_v, f_v):
    assert len(t_v) == len(f_v), 't_v must have same length as f_v'
    assert all(t_i < t_j for t_i,t_j in zip(t_v[:-1],t_v[1:])), 't_v must be strictly increasing'
    assert len(f_v) == 3, 'require 3 values and the middle value must be a local min'
    assert f_v[0] > f_v[1] and f_v[1] < f_v[2]

    # Solve for the coefficients of the quadratic q(t) := q_v[0]*t^0 + q_v[1]*t^1 + q_v[2]*t^2
    # using the specified time values t_v and function values f_v;
    #     q(t_v[0]) = f_v[0]
    #     q(t_v[1]) = f_v[1]
    #     q(t_v[2]) = f_v[2]
    T = np.array([polynomial_basis_vector(t,2) for t in t_v])
    q_v = np.linalg.solve(T, f_v)
    assert np.allclose(np.dot(T, q_v), f_v), 'q_v failed to solve intended equation T*q_v == f_v'
    # This is the critical point of q.
    t_min = -0.5*q_v[1]/q_v[2]
    assert t_v[0] < t_min < t_v[-1], 't_min should be bounded by first and last t_v values because f_v[1] is a local min.'
    # This is the corresponding value q(t_min)
    q_min = np.dot(polynomial_basis_vector(t_min,2), q_v)

    return t_min,q_min

class DynamicsContext(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def configuration_space_dimension (cls):
        pass

    @classmethod
    @abc.abstractmethod
    def H (cls, qp):
        """Evaluates the Hamiltonian on the (2,N)-shaped (q,p) coordinate."""
        pass

    @classmethod
    @abc.abstractmethod
    def dH_dq (cls, q, p):
        """Evaluates the partial of H with respect to q on the (2,N)-shaped (q,p) coordinate.  Returns a (N,)-vector."""
        pass

    @classmethod
    @abc.abstractmethod
    def dH_dp (cls, q, p):
        """Evaluates the partial of H with respect to q on the (2,N)-shaped (q,p) coordinate.  Returns a (N,)-vector."""
        pass

    @classmethod
    def X_H (cls, coordinates):
        """
        Computes the Hamiltonian vector field on the (q,p) coordinate reshaped as a (2*N,)-vector, returns the same shape.

        \omega^-1 * dH (i.e. the symplectic gradient of H) is the hamiltonian vector field for this system.
        X is the list of coordinates [x, y, z, p_x, p_y, p_z].
        t is the time at which to evaluate the flow.  This particular vector field is independent of time.

        If the tautological one-form on the cotangent bundle is

            tau := p dq

        then the symplectic form is

            omega := -dtau = -dq wedge dp

        which, in the coordinates (q_0, q_1, p_0, p_1), has the matrix

            [  0  0 -1  0 ]
            [  0  0  0 -1 ]
            [  1  0  0  0 ]
            [  0  1  0  0 ],

        or in matrix notation, with I denoting the 2x2 identity matrix,

            [  0 -I ]
            [  I  0 ],

        having inverse

            [  0  I ]
            [ -I  0 ].

        With dH:

            dH = dH/dq * dq + dH/dp * dp,    (here, dH/dq denotes the partial of H w.r.t. q)

        or expressed in coordinates as

            [ dH/dq ]
            [ dH/dp ]

        it follows that the sympletic gradient of H is

            dH/dp * dq - dH/dq * dp

        or expressed in coordinates as

            [  dH/dp ]
            [ -dH/dq ],

        which is Hamilton's equations.
        """
        q = coordinates[0,:]
        p = coordinates[1,:]
        # This is the symplectic gradient of H.
        return np.concatenate((cls.dH_dp(q,p), -cls.dH_dq(q,p)))

    @classmethod
    def phase_space_dimension (cls):
        return 2*cls.configuration_space_dimension()

class HeisenbergDynamicsContext(DynamicsContext):
    @classmethod
    def configuration_space_dimension (cls):
        return 3

    @classmethod
    def K (cls, qp):
        # If qp is denoted as ((x,y,z),(p_x,p_y,p_z)),
        # P_x := p_x - 0.5*y*p_z,
        # P_y := p_y + 0.5*x*p_z,
        # then
        # K = 0.5*(P_x**2 + P_y**2)
        P_x = qp[1,0] - qp[0,1]*qp[1,2]/2
        P_y = qp[1,1] + qp[0,0]*qp[1,2]/2
        return (P_x**2 + P_y**2)/2

    @classmethod
    def V (cls, qp):
        # If qp is denoted as ((x,y,z),(p_x,p_y,p_z)),
        # r_squared := x**2 + y**2,
        # mu        := r_squared**2 + beta*z**2
        # then
        # V = -alpha / sqrt(mu)
        r_squared = qp[0,0]**2 + qp[0,1]**2
        mu        = r_squared**2 + cls.beta()*qp[0,2]**2
        return -cls.alpha() / cls.sqrt(mu)

    @classmethod
    def H (cls, qp):
        return cls.K(qp) + cls.V(qp)

    @classmethod
    def J (cls, qp):
        return qp[0,0]*qp[1,0] + qp[0,1]*qp[1,1] + 2*qp[0,2]*qp[1,2]

    @classmethod
    def dH_dq (cls, q, p):
        assert np.all(np.isfinite(q)), 'encountered non-finite q: {0}. p: {1}'.format(q, p)
        assert np.all(np.isfinite(p)), 'encountered non-finite p: {0}. q: {1}'.format(p, q)
        P_x = p[0] - q[1]*p[2]/2
        P_y = p[1] + q[0]*p[2]/2
        r_squared = q[0]**2 + q[1]**2
        mu = r_squared**2 + cls.beta()*q[2]**2
        alpha_times_mu_to_neg_three_halves = cls.alpha() / cls.sqrt(mu)**3
        return np.array((
             P_y*p[2]/2 + alpha_times_mu_to_neg_three_halves*r_squared*2*q[0],
            -P_x*p[2]/2 + alpha_times_mu_to_neg_three_halves*r_squared*2*q[1],
             cls.beta()*alpha_times_mu_to_neg_three_halves*q[2]
        ))

    @classmethod
    def dH_dp (cls, q, p):
        P_x = p[0] - q[1]*p[2]/2
        P_y = p[1] + q[0]*p[2]/2
        #print('q = {0}, p = {1}, P_x = {1}, P_y = {2}, P_z = {3}'.format(q, p, P_x, P_y, (q[0]*P_y - q[1]*P_x)/2))
        return np.array((
            P_x,
            P_y,
            (q[0]*P_y - q[1]*P_x)/2
        ))

    @classmethod
    @abc.abstractmethod
    def sqrt (cls, x):
        """Compute the sqrt of x."""
        pass

    @classmethod
    @abc.abstractmethod
    def pi (cls):
        """Return the value pi."""
        pass

    @classmethod
    def alpha (cls):
        """Return the alpha value occurring in the fundamental solution to the sub-Riemannian Laplacian."""
        #return 2/cls.pi()
        return cls.pi()/8

    @classmethod
    def beta (cls):
        """Return the beta value occurring in the fundamental solution to the sub-Riemannian Laplacian."""
        return 16

class HeisenbergDynamicsContext_Symbolic(HeisenbergDynamicsContext):
    @classmethod
    def sqrt (cls, x):
        return sp.sqrt(x)

    @classmethod
    def pi (cls):
        return sp.pi

    @classmethod
    def initial_condition (cls):
        # Symbolically solve H(1,0,0,0,1,p_z) = 0 for p_z.
        p_z = sp.var('p_z')
        zero = sp.Integer(0)
        one = sp.Integer(1)
        # H = HeisenbergDynamicsContext.hamiltonian(np.array([one, zero, zero, zero, one, p_z], dtype=object), sqrt=sp.sqrt, pi=sp.pi)
        H = cls.H(
            np.array(
                (
                    (one/2, zero, zero),
                    ( zero,  one,  p_z)
                ),
                dtype=object
            )
        )
        print('H = {0}'.format(H))
        p_z_solution = np.max(sp.solve(H, p_z))
        print('p_z = {0}'.format(p_z_solution))
        p_z_solution = float(p_z_solution)
        # TODO: Somehow subs into H (symbolic expression) and evaluate to float
        return np.array((
            (0.5, 0.0,          0.0),
            (0.0, 1.0, p_z_solution)
        ))

class HeisenbergDynamicsContext_Numeric(HeisenbergDynamicsContext):
    @classmethod
    def sqrt (cls, x):
        return np.sqrt(x)

    @classmethod
    def pi (cls):
        return np.pi

    @classmethod
    def initial_condition_preimage (cls):
        #return np.array((0.46200237, 0.0, 0.97966453))
        #return np.array((4.62385150336783013e-01, -5.02075714050898860e-04, 9.80340082913902178e-01))

        # from
        # qp_opt = [[4.62167379391418609e-01 0.00000000000000000e+00 0.00000000000000000e+00]
        #           [-4.67440934052728782e-04 9.80312987653756296e-01 6.32317054716479721e+00]]
        return np.array((4.62167379391418609e-01, -4.67440934052728782e-04, 9.80312987653756296e-01))

    def __solve_for_embedding2 (self):
        # Symbolically solve H(qp) = 0 for qp[1,2].
        X = vorpy.symbolic.tensor('X', (3,))
        zero = sp.Integer(0)
        one = sp.Integer(1)
        qp = np.array(
            (
                ( one, zero, zero),
                (X[0], X[1], X[2]),
            ),
            dtype=object
        )
        H = HeisenbergDynamicsContext_Symbolic.H(qp)
        #print('H(qp) = {0}'.format(H))
        p_z = qp[1,2] # Momentum for z coordinate
        p_z_solution_v = sp.solve(H, p_z)
        #print('len(p_z_solution_v) = {0}'.format(len(p_z_solution_v)))
        #print('p_z_solution_v = {0}'.format(p_z_solution_v))
        # Just take the last solution.
        p_z_solution = p_z_solution_v[-1]
        print('p_z_solution = {0}'.format(p_z_solution))
        # The domain for this function is
        #     -sqrt(pi/4) <= p_x <= sqrt(pi/4)

        self.symbolic_embedding2_domain = X[:2]
        self.symbolic_embedding2 = np.array(
            (
                ( one, zero,         zero),
                (X[0], X[1], p_z_solution),
            ),
            dtype=object
        )
        self.embedding2 = vorpy.symbolic.lambdified(
            self.symbolic_embedding2,
            self.symbolic_embedding2_domain,
            replacement_d={
                'array'         :'np.array',
                'ndarray'       :'np.ndarray',
                'dtype=object'  :'dtype=float',
                'sqrt'          :'np.sqrt',
                'pi'            :'np.pi',
            },
            #verbose=True
        )

    def __solve_for_embedding3 (self):
        # Symbolically solve H(qp) = 0 for qp[1,2].
        X = vorpy.symbolic.tensor('X', (4,))
        zero = sp.Integer(0)
        qp = np.array(
            (
                (X[0], zero, zero),
                (X[1], X[2], X[3]),
            ),
            dtype=object
        )
        H = HeisenbergDynamicsContext_Symbolic.H(qp)
        #print('H(qp) = {0}'.format(H))
        p_z = qp[1,2] # Momentum for z coordinate
        p_z_solution_v = sp.solve(H, p_z)
        #print('len(p_z_solution_v) = {0}'.format(len(p_z_solution_v)))
        #print('p_z_solution_v = {0}'.format(p_z_solution_v))
        # Just take the last solution.
        p_z_solution = p_z_solution_v[-1]
        #print('p_z_solution = {0}'.format(p_z_solution))

        self.symbolic_embedding3_domain = X[:3]
        self.symbolic_embedding3 = np.array(
            (
                (X[0], zero,         zero),
                (X[1], X[2], p_z_solution),
            ),
            dtype=object
        )
        self.embedding3 = vorpy.symbolic.lambdified(
            self.symbolic_embedding3,
            self.symbolic_embedding3_domain,
            replacement_d={
                'array'         :'np.array',
                'ndarray'       :'np.ndarray',
                'dtype=object'  :'dtype=float',
                'sqrt'          :'np.sqrt',
                'pi'            :'np.pi',
            },
            #verbose=True
        )

    def __init__ (self):
        self.__solve_for_embedding2()
        self.__solve_for_embedding3()

    #@classmethod
    #def initial_condition_preimage (cls):
        ##return np.array((0.5, 1.0)) # original
        ##return np.array((0.46307038, 0.9807273)) # once optimized
        ##return np.array((0.4613605, 0.98053092)) # twice optimized
        #return np.array((0.46200237, 0.97966453)) # thrice optimized

    #def __init__ (self):
        ## Symbolically solve H(qp) = 0 for qp[1,2].
        #X = vorpy.symbolic.tensor('X', (3,))
        #zero = sp.Integer(0)
        #qp = np.array(
            #(
                #(X[0], zero, zero),
                #(zero, X[1], X[2]),
            #),
            #dtype=object
        #)
        #H = HeisenbergDynamicsContext_Symbolic.H(qp)
        #print('H(qp) = {0}'.format(H))
        #p_z = qp[1,2] # Momentum for z coordinate
        #p_z_solution_v = sp.solve(H, p_z)
        #print('len(p_z_solution_v) = {0}'.format(len(p_z_solution_v)))
        ##print('p_z_solution_v = {0}'.format(p_z_solution_v))
        ## Just take the last solution.
        #p_z_solution = p_z_solution_v[-1]
        ##print('p_z_solution = {0}'.format(p_z_solution))

        #self.symbolic_embedding_domain = X[:2]
        #self.symbolic_embedding = np.array(
            #(
                #(X[0], zero,         zero),
                #(zero, X[1], p_z_solution),
            #),
            #dtype=object
        #)
        #self.embedding = vorpy.symbolic.lambdified(
            #self.symbolic_embedding,
            #self.symbolic_embedding_domain,
            #replacement_d={
                #'array'         :'np.array',
                #'ndarray'       :'np.ndarray',
                #'dtype=object'  :'dtype=float',
                #'sqrt'          :'np.sqrt',
                #'pi'            :'np.pi',
            #},
            #verbose=True
        #)

    #@classmethod
    #def initial_condition_preimage (cls):
        ##return np.array((0.5, 0.0, 0.0, 0.0, 1.0)) # original
        #return np.array((5.00647217e-01, 1.02238132e-03, -2.18960185e-04, 1.39439904e-03, 9.99489776e-01))

    #def __init__ (self):
        ## Symbolically solve H(qp) = 0 for qp[1,2].
        #X = vorpy.symbolic.tensor('X', (6,))
        #qp = X.reshape(2,3)
        #H = HeisenbergDynamicsContext_Symbolic.H(qp)
        #print('H(qp) = {0}'.format(H))
        #p_z = qp[1,2] # Momentum for z coordinate
        #p_z_solution_v = sp.solve(H, p_z)
        #print('len(p_z_solution_v) = {0}'.format(len(p_z_solution_v)))
        ##print('p_z_solution_v = {0}'.format(p_z_solution_v))
        ## Just take the last solution.
        #p_z_solution = p_z_solution_v[-1]
        ##print('p_z_solution = {0}'.format(p_z_solution))

        #self.symbolic_embedding_domain = X[:5]
        #self.symbolic_embedding = np.array(
            #(
                #(X[0], X[1],         X[2]),
                #(X[3], X[4], p_z_solution),
            #),
            #dtype=object
        #)
        #self.embedding = vorpy.symbolic.lambdified(
            #self.symbolic_embedding,
            #self.symbolic_embedding_domain,
            #replacement_d={
                #'array'         :'np.array',
                #'ndarray'       :'np.ndarray',
                #'dtype=object'  :'dtype=float',
                #'sqrt'          :'np.sqrt',
                #'pi'            :'np.pi',
            #},
            #verbose=True
        #)

class ShootingMethodObjective:
    def __init__ (self, *, dynamics_context, qp_0, t_max, t_delta):
        self.__dynamics_context         = dynamics_context
        self.qp_0                       = qp_0
        self.__t_v                      = None
        self.__qp_v                     = None
        self.t_max                      = t_max
        self.t_delta                    = t_delta
        self.__Q_v                      = None
        self.__Q_global_min_index       = None
        self.__t_min                    = None
        self.__objective                = None
        self.flow_curve_was_salvaged    = False

    def configuration_space_dimension (self):
        return self.__dynamics_context.configuration_space_dimension()

    def flow_curve (self):
        if self.__qp_v is None:
            start_time = time.time() # TODO: Replace with Ticker usage

            t_v = np.arange(0.0, self.t_max, self.t_delta)
            order = 2
            #omega = vorpy.symplectic_integration.nonseparable_hamiltonian.heuristic_estimate_for_omega(delta=self.t_delta, order=order, c=10.0)
            # Want 2*omega*t_delta = pi/2, meaning that omega = pi/(4*t_delta)
            omega = np.pi/(4*self.t_delta)
            assert np.allclose(2*omega*self.t_delta, np.pi/2)
            try:
                qp_v = vorpy.symplectic_integration.nonseparable_hamiltonian.integrate(
                    initial_coordinates=self.qp_0,
                    t_v=t_v,
                    dH_dq=self.__dynamics_context.dH_dq,
                    dH_dp=self.__dynamics_context.dH_dp,
                    order=order,
                    omega=omega
                )
                self.flow_curve_was_salvaged = False
            except vorpy.symplectic_integration.exceptions.SalvagedResultException as e:
                print('salvaged results from exception encountered in nonseparable_hamiltonian.integrate: {0}'.format(e))
                original_step_count = len(t_v)
                self.__qp_v = qp_v  = e.integrated_coordinates[:e.salvaged_step_count,...]
                self.__t_v  = t_v   = t_v[:e.salvaged_step_count]
                self.flow_curve_was_salvaged = True

                # TEMP: Plot this salvaged curve in order to diagnose what went wrong
                orbit_plot = OrbitPlot(row_count=1, extra_col_count=0)
                orbit_plot.plot_curve(
                    curve_description='salvaged curve - {0} steps out of {1}'.format(e.salvaged_step_count, original_step_count),
                    axis_v=orbit_plot.axis_vv[0],
                    smo=self
                )
                orbit_plot.plot_and_clear(
                    filename=os.path.join(
                        'shooting_method_3.custom_plot',
                        'salvaged.obj:{0:.4e}.t_delta:{1:.3e}.t_max:{2:.3e}.ic:{3}.png'.format(
                            self.objective(),
                            self.t_delta,
                            self.t_max,
                            ndarray_as_single_line_string(self.qp_0)
                        )
                    )
                )

            print('integration took {0} seconds'.format(time.time() - start_time))

            self.__t_v = t_v
            self.__qp_v = qp_v
        return self.__qp_v

    def t_v (self):
        if self.__t_v is None:
            self.flow_curve()
            assert self.__t_v is not None
        return self.__t_v

    def squared_distance_function (self):
        if self.__Q_v is None:
            # Let s denote squared distance function s(t) := 1/2 |qp_0 - flow_of_qp_0(t))|^2
            #self.__Q_v = 0.5 * np.sum(np.square(self.flow_curve() - self.qp_0), axis=-1)
            self.__Q_v = vorpy.apply_along_axes(lambda x:0.5*np.sum(np.square(x)), (-2,-1), (self.flow_curve() - self.qp_0,), output_axis_v=(), func_output_shape=())
        return self.__Q_v

    def t_min (self):
        if self.__t_min is None:
            self.compute_t_min_and_objective()
        return self.__t_min

    def objective (self):
        if self.__objective is None:
            self.compute_t_min_and_objective()
        return self.__objective

    def Q_global_min_index (self):
        if self.__Q_global_min_index is None:
            self.compute_t_min_and_objective()
        return self.__Q_global_min_index

    def __call__ (self):
        return self.objective()

    def compute_t_min_and_objective (self):
        t_v                             = self.t_v()
        Q_v                             = self.squared_distance_function()

        local_min_index_v               = [i for i in range(1,len(Q_v)-1) if Q_v[i-1] > Q_v[i] and Q_v[i] < Q_v[i+1]]
        Q_local_min_v                   = [Q_v[i] for i in local_min_index_v]
        try:
            Q_local_min_min_index       = np.argmin(Q_local_min_v)
            self.__Q_global_min_index   = _Q_global_min_index = local_min_index_v[Q_local_min_min_index]
            if True:
                # Fit a quadratic function to the 3 points centered on the argmin in order to have
                # sub-sample accuracy when calculating the objective function value.
                assert 1 <= _Q_global_min_index < len(Q_v)-1
                s                       = slice(_Q_global_min_index-1, _Q_global_min_index+2)
                self.__t_min,self.__objective = quadratic_min_time_parameterized(t_v[s], Q_v[s])
                # Some tests show this discrepancy to be on the order of 1.0e-9
                #print('self.__objective - Q_v[_Q_global_min_index] = {0}'.format(self.__objective - Q_v[_Q_global_min_index]))
            else:
                self.__t_min            = t_v[_Q_global_min_index]
                self.__objective        = Q_v[_Q_global_min_index]
        except ValueError:
            # If there was no local min, then declare the objective function value to be NaN
            self.__Q_global_min_index   = None
            self.__t_min                = np.nan
            self.__objective            = np.nan

def evaluate_shooting_method_objective (dynamics_context, qp_0, t_max, t_delta):
    return ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_0, t_max=t_max, t_delta=t_delta)()

class OrbitPlot:
    def __init__ (self, *, row_count, extra_col_count):
        row_count = row_count
        col_count = 5+extra_col_count
        self.fig,self.axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(15*col_count,15*row_count))

    def plot_curve (self, *, curve_description, axis_v, smo):
        flow_curve = smo.flow_curve()
        H_v = vorpy.apply_along_axes(HeisenbergDynamicsContext_Numeric.H, (-2,-1), (flow_curve,), output_axis_v=(), func_output_shape=())
        abs_H_v = np.abs(H_v)

        axis_index = 0

        axis = axis_v[axis_index]
        axis.set_title('{0} curve xy-position'.format(curve_description))
        axis.plot(0, 0, 'o', color='black')
        axis.plot(flow_curve[:,0,0], flow_curve[:,0,1])
        axis.plot(flow_curve[0,0,0], flow_curve[0,0,1], 'o', color='green', alpha=0.5)
        # TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
        axis.plot(flow_curve[smo.Q_global_min_index(),0,0], flow_curve[smo.Q_global_min_index(),0,1], 'o', color='red', alpha=0.5)
        axis.set_aspect('equal')
        axis_index += 1

        axis = axis_v[axis_index]
        axis.set_title('{0} curve z-position (green line indicates t_min)'.format(curve_description))
        axis.axhline(0, color='black')
        axis.plot(smo.t_v(), flow_curve[:,0,2])
        axis.axvline(smo.t_min(), color='green')
        axis_index += 1

        #axis = axis_v[axis_index]
        #axis.set_title('{0} curve xy-momentum'.format(curve_description))
        #axis.plot(flow_curve[:,1,0], flow_curve[:,1,1])
        #axis.plot(flow_curve[0,1,0], flow_curve[0,1,1], 'o', color='green', alpha=0.5)
        ## TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
        #axis.plot(flow_curve[smo.Q_global_min_index(),1,0], flow_curve[smo.Q_global_min_index(),1,1], 'o', color='red', alpha=0.5)
        #axis.set_aspect('equal')
        #axis_index += 1

        #axis = axis_v[axis_index]
        #axis.set_title('{0} curve z-momentum'.format(curve_description))
        #axis.axhline(0, color='black')
        #axis.plot(smo.t_v(), flow_curve[:,1,2])
        #axis.axvline(smo.t_min(), color='green')
        #axis_index += 1

        axis = axis_v[axis_index]
        axis.set_title('abs(H) (should stay close to 0)\nmax(abs(H)) = {0:.2e}, H_0 = {1:e}'.format(np.max(abs_H_v), H_v[0]))
        axis.semilogy(smo.t_v(), abs_H_v)
        axis_index += 1

        J_v = vorpy.apply_along_axes(HeisenbergDynamicsContext_Numeric.J, (-2,-1), (flow_curve,), output_axis_v=(), func_output_shape=())
        J_0 = J_v[0]
        J_v -= J_0
        abs_J_minus_J_0 = np.abs(J_v)

        axis = axis_v[axis_index]
        axis.set_title('abs(J - J_0) (should be close to 0)\nJ_0 = {0}; max(abs(J - J_0)) = {1:.2e}'.format(J_0, np.max(abs_J_minus_J_0)))
        axis.semilogy(smo.t_v(), abs_J_minus_J_0)
        axis_index += 1

        axis = axis_v[axis_index]
        axis.set_title('squared distance to initial condition\nt_min = {0}, min sqd = {1:.17e}'.format(smo.t_min(), smo.objective()))
        axis.semilogy(smo.t_v(), smo.squared_distance_function())
        axis.axvline(smo.t_min(), color='green')
        axis_index += 1

    def plot_and_clear (self, *, filename):
        self.fig.tight_layout()
        plt.savefig(filename)
        print('wrote to file "{0}"'.format(filename))
        # VERY important to do this -- otherwise your memory will slowly fill up!
        # Not sure which one is actually sufficient
        plt.clf()
        plt.close(self.fig)
        del self.fig
        del self.axis_vv

import optparse

class OptionParser:
    def __init__ (self):
        self.op = optparse.OptionParser()
        self.op.add_option(
            '--optimization-iterations',
            dest='optimization_iterations',
            default=1000,
            type='int',
            help='Specifies the number of iterations to run the optimization for (if applicable).  Default is 1000.'
        )
        self.op.add_option(
            '--dt',
            dest='dt',
            default='0.001',
            type='float',
            help='Specifies the timestep for the curve integration.'
        )
        self.op.add_option(
            '--max-time',
            dest='max_time',
            default='50.0',
            type='float',
            help='Specifies the max time to integrate the curve to.'
        )
        self.op.add_option(
            '--initial-2preimage',
            dest='initial_2preimage',
            type='string',
            help='Specifies the preimage of the initial conditions with respect to the [p_x,p_y] |-> [[x,y,z],[p_x,p_y,p_z]] embedding.  Should have the form [p_x,p_y], where each of p_x,p_y are floating point literals.'
        )
        self.op.add_option(
            '--initial-3preimage',
            dest='initial_3preimage',
            type='string',
            help='Specifies the preimage of the initial conditions with respect to the [x,p_x,p_y] |-> [[x,y,z],[p_x,p_y,p_z]] embedding.  Should have the form [x,p_x,p_y], where each of x,y,z are floating point literals.'
        )
        self.op.add_option(
            '--initial',
            dest='initial',
            type='string',
            help='Specifies the initial conditions [x,y,z,p_x,p_y,p_z], where each of x,y,z,p_x,p_y,p_z are floating point literals.'
        )
        self.op.add_option(
            '--seed',
            dest='seed',
            default=666,
            type='int',
            help='Specifies the seed to use for pseudorandom number generation.  Using the same seed should produce the same sequence of random numbers, and therefore provide reproducible program execution.'
        )
        self.op.add_option(
            '--optimize-initial',
            dest='optimize_initial',
            action='store_true',
            default=False,
            help='Indicates that the specified initial condition (via whichever of the --initial... options) should be used as the starting point for an optimization to attempt to close the orbit.  Default value is False.'
        )
        self.op.add_option(
            '--search',
            dest='search',
            action='store_true',
            default=False,
            help='Indicates that random initial conditions should be generated and for each, if a threshold is met, an optimization routine run to attempt to close the orbit.'
        )
        self.op.add_option(
            '--k-fold-initial',
            dest='k',
            type='int',
            help='Specifies that the given value, call it k, should be used in a particular form of initial condition intended to produce a k-fold symmetric orbit -- experimental.'
        )
        self.op.add_option(
            '--abortive-threshold',
            dest='abortive_threshold',
            default=0.1,
            type='float',
            help='Sets the threshold below which a candidate curve\'s objective function will qualify it for running through the optimizer to attempt to close it.'
        )

    @staticmethod
    def __pop_brackets_off_of (string):
        if len(string) < 2:
            raise ValueError('string (which is "{0}") must be at least 2 chars long'.format(string))
        elif string[0] != '[' or string[-1] != ']':
            raise ValueError('string (which is "{0}") must begin with [ and end with ]'.format(string))
        return string[1:-1]

    @staticmethod
    def __csv_as_ndarray (string, dtype):
        return np.array([dtype(token) for token in string.split(',')])

    def parse_argv_and_validate (self, argv, dynamics_context):
        options,args = self.op.parse_args()

        if options.search:
            require_initial_conditions = False
        elif options.k is not None:
            require_initial_conditions = False
            options.qp_0 = np.array([
                [1.0,             0.0, 0.25*np.sqrt(options.k**4 * np.pi**2 * 0.0625 - 1.0)],
                [0.0, 1.0 / options.k,                                                  0.0]
            ])
        else:
            require_initial_conditions = True

        num_initial_conditions_specified = sum([
            options.initial_2preimage is not None,
            options.initial_3preimage is not None,
            options.initial is not None
        ])
        if require_initial_conditions:
            if num_initial_conditions_specified != 1:
                print('if neither --search nor --k-fold-initial are not specified, then you must specify exactly one of --initial-2preimage or --initial-3preimage or --initial, but {0} of those were specified.'.format(num_initial_conditions_specified))
                self.op.print_help()
                return None,None

        if options.dt is None:
            print('required option --dt was not specified.')
            self.op.print_help()
            return None,None

        if options.max_time is None:
            print('required option --max-time was not specified.')
            self.op.print_help()
            return None,None

        if require_initial_conditions:
            # Attempt to parse initial conditions.  Upon success, the attribute options.qp_0 should exist.
            if options.initial_2preimage is not None:
                try:
                    options.initial_2preimage = OptionParser.__csv_as_ndarray(OptionParser.__pop_brackets_off_of(options.initial_2preimage), float)
                    expected_shape = (2,)
                    if options.initial_2preimage.shape != expected_shape:
                        raise ValueError('--initial_2preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_2preimage.shape, expected_shape))
                    options.qp_0 = dynamics_context.embedding2(options.initial_2preimage)
                except ValueError as e:
                    print('error parsing --initial_2preimage value: {0}'.format(str(e)))
                    self.op.print_help()
                    return None,None
            elif options.initial_3preimage is not None:
                try:
                    options.initial_3preimage = OptionParser.__csv_as_ndarray(OptionParser.__pop_brackets_off_of(options.initial_3preimage), float)
                    expected_shape = (3,)
                    if options.initial_3preimage.shape != expected_shape:
                        raise ValueError('--initial_3preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_3preimage.shape, expected_shape))
                    options.qp_0 = dynamics_context.embedding3(options.initial_3preimage)
                except ValueError as e:
                    print('error parsing --initial_3preimage value: {0}'.format(str(e)))
                    self.op.print_help()
                    return None,None
            elif options.initial is not None:
                try:
                    options.initial = OptionParser.__csv_as_ndarray(OptionParser.__pop_brackets_off_of(options.initial), float)
                    expected_shape = (6,)
                    if options.initial.shape != expected_shape:
                        raise ValueError('--initial value had the wrong number of components (got {0} but expected {1}).'.format(options.initial.shape, expected_shape))
                    options.qp_0 = options.initial.reshape(2,3)
                except ValueError as e:
                    print('error parsing --initial value: {0}'.format(str(e)))
                    self.op.print_help()
                    return None,None
            else:
                assert False, 'this should never happen because of the check with num_initial_conditions_specified'

        return options,args

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

def construct_filename (*, obj, t_delta, t_max, initial_condition, t_min):
    return 'obj:{0:.4e}.t_delta:{1:.3e}.t_max:{2:.3e}.initial_condition:{3}.t_min:{4:.17e}.png'.format(obj, t_delta, t_max, ndarray_as_single_line_string(initial_condition), t_min)

def search (dynamics_context, options):
    if not os.path.exists('shooting_method_3/'):
        os.mkdir('shooting_method_3')

    if not os.path.exists('shooting_method_3/abortive'):
        os.mkdir('shooting_method_3/abortive')

    np.set_printoptions(formatter={'float':float_formatter})

    def try_random_initial_condition ():
        ##X_0 = rng.randn(*HeisenbergDynamicsContext_Numeric.initial_condition_preimage().shape)
        #X_0 = rng.randn(2)
        ## NOTE: This somewhat biases the generation of random initial conditions
        ##X_0[0] = np.exp(X_0[0]) # So we never get negative values
        #X_0[1] = np.abs(X_0[1]) # So we only bother pointing upward

        # NOTE: The goal here is to sample uniformly over the domain:
        #     -sqrt(pi/4) <= p_x <= sqrt(pi/4)
        #               0 <= p_y <= C
        # for some arbitrary positive bound C, say 2.

        C = 2.0
        epsilon = 1.0e-5
        # Perturb the bounds for p_x by epsilon away from the actual bound.
        X_0 = np.array([
            rng.uniform(-np.sqrt(np.pi/4)+epsilon, np.sqrt(np.pi/4)-epsilon),
            rng.uniform(0.0, C)
        ])

        #X_0 = np.array([4.53918797113298744e-01,-6.06738228528062038e-04,1.75369725636529949e+00])

        qp_0 = dynamics_context.embedding2(X_0)
        print('randomly generated initial condition preimage: X_0:')
        print(X_0)
        #print('embedding of randomly generated initial condition preimage: qp_0:')
        #print(qp_0)
        t_max = 5.0
        # TODO: Pick a large-ish t_max, then cluster the local mins, and then from the lowest cluster,
        # pick the corresponding to the lowest time value, and then make t_max 15% larger than that.
        while True:
            smo_0 = ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_0, t_max=t_max, t_delta=options.dt)
            print('smo_0.objective() = {0:.17e}, smo.t_min() = {1}'.format(smo_0.objective(), smo_0.t_min()))
            if smo_0.objective() < options.abortive_threshold:
                break
            else:
                t_max *= 1.5
                print('curve did not nearly close up -- retrying with higher t_max: {0}'.format(t_max))
            if t_max > options.max_time:
                print('t_max ({0}) was raised too many times, exceeding --max-time value of {1}, before nearly closing up -- aborting'.format(t_max, options.max_time))

                orbit_plot = OrbitPlot(row_count=1, extra_col_count=0)
                orbit_plot.plot_curve(curve_description='initial', axis_v=orbit_plot.axis_vv[0], smo=smo_0)
                orbit_plot.plot_and_clear(
                    filename=os.path.join(
                        'shooting_method_3/abortive',
                        construct_filename(
                            obj=smo_0.objective(),
                            t_delta=options.dt,
                            t_max=t_max,
                            initial_condition=qp_0,
                            t_min=smo_0.t_min()
                        )
                    )
                )

                return
        flow_curve_0 = smo_0.flow_curve()

        optimizer = library.monte_carlo.MonteCarlo(
            obj=lambda qp_0:evaluate_shooting_method_objective(dynamics_context, qp_0, t_max, options.dt),
            initial_parameters=X_0,
            inner_radius=1.0e-12,
            outer_radius=1.0e-1,
            rng_seed=options.seed,
            embedding=dynamics_context.embedding2
        )
        try:
            actual_iteration_count = 0
            for i in range(options.optimization_iterations):
                optimizer.compute_next_step()
                actual_iteration_count += 1
                print('i = {0}, obj = {1:.17e}'.format(i, optimizer.obj_history_v[-1]))
        except KeyboardInterrupt:
            print('got KeyboardInterrupt -- halting optimization, but will still plot current results')
        except AssertionError:
            print('got AssertionError -- halting optimization, but will plot last good results')

        qp_opt = optimizer.embedded_parameter_history_v[-1]
        smo_opt = ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_opt, t_max=t_max, t_delta=options.dt)
        flow_curve_opt = smo_opt.flow_curve()

        print('qp_opt = {0}'.format(qp_opt))
        print('qp_opt embedding preimage; X_0 = {0}'.format(optimizer.parameter_history_v[-1]))

        orbit_plot = OrbitPlot(row_count=2, extra_col_count=1)

        orbit_plot.plot_curve(curve_description='initial', axis_v=orbit_plot.axis_vv[0], smo=smo_0)
        orbit_plot.plot_curve(curve_description='optimized', axis_v=orbit_plot.axis_vv[1], smo=smo_opt)

        axis = orbit_plot.axis_vv[0][-1]
        axis.set_title('objective function history')
        axis.semilogy(optimizer.obj_history_v)

        orbit_plot.plot_and_clear(
            filename=os.path.join(
                'shooting_method_3',
                construct_filename(
                    obj=smo_opt.objective(),
                    t_delta=options.dt,
                    t_max=t_max,
                    initial_condition=qp_opt,
                    t_min=smo_opt.t_min()
                )
            )
        )

    try:
        while True:
            try:
                try_random_initial_condition()
            except Exception as e:
                print('encountered exception during try_random_initial_condition; skipping.  exception was: {0}'.format(e))
                pass
    except KeyboardInterrupt:
        print('got KeyboardInterrupt -- exiting program')
        sys.exit(0)


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    import sys

    # https://github.com/matplotlib/matplotlib/issues/5907 says this should fix "Exceeded cell block limit" problems
    matplotlib.rcParams['agg.path.chunksize'] = 10000

    dynamics_context = HeisenbergDynamicsContext_Numeric()

    option_parser = OptionParser()
    options,args = option_parser.parse_argv_and_validate(sys.argv, dynamics_context)
    if options is None:
        sys.exit(-1)

    print('options: {0}'.format(options))
    print('args   : {0}'.format(args))

    rng = np.random.RandomState(options.seed)

    if options.search:
        search(dynamics_context, options)
    else:
        if not os.path.exists('shooting_method_3.custom_plot/'):
            os.mkdir('shooting_method_3.custom_plot')

        # Plot given curve
        qp_0 = options.qp_0
        smo_0 = ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=options.qp_0, t_max=options.max_time, t_delta=options.dt)
        print('smo_0.objective() = {0}'.format(smo_0.objective()))

        if options.optimize_initial:
            if options.initial_2preimage is not None:
                X_0 = options.initial_2preimage
                embedding = dynamics_context.embedding2
            elif options.initial_3preimage is not None:
                X_0 = options.initial_2preimage
                embedding = dynamics_context.embedding3
            elif options.initial is not None:
                X_0 = options.qp_0
                embedding = None
            else:
                assert options.k is not None
                X_0 = options.qp_0
                embedding = None

            optimizer = library.monte_carlo.MonteCarlo(
                obj=lambda qp_0:evaluate_shooting_method_objective(dynamics_context, qp_0, options.max_time, options.dt),
                initial_parameters=X_0,
                inner_radius=1.0e-12,
                outer_radius=1.0e-1,
                rng_seed=options.seed, # TODO: Make a single RNG that's used everywhere in the program.
                embedding=embedding
            )
            try:
                actual_iteration_count = 0
                for i in range(options.optimization_iterations):
                    optimizer.compute_next_step()
                    actual_iteration_count += 1
                    print('i = {0}, obj = {1:.17e}'.format(i, optimizer.obj_history_v[-1]))
            except KeyboardInterrupt:
                print('got KeyboardInterrupt -- halting optimization, but will still plot current results')
            except AssertionError:
                print('got AssertionError -- halting optimization, but will plot last good results')

            qp_opt = optimizer.embedded_parameter_history_v[-1]
            smo_opt = ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_opt, t_max=options.max_time, t_delta=options.dt)

            print('qp_opt = {0}'.format(qp_opt))
            if embedding is not None:
                print('qp_opt embedding preimage; X_0 = {0}'.format(optimizer.parameter_history_v[-1]))

            orbit_plot = OrbitPlot(row_count=2, extra_col_count=1)
            orbit_plot.plot_curve(curve_description='optimized', axis_v=orbit_plot.axis_vv[1], smo=smo_opt)

            axis = orbit_plot.axis_vv[0][-1]
            axis.set_title('objective function history')
            axis.semilogy(optimizer.obj_history_v)

            qp = qp_opt
            smo = smo_opt
        else:
            orbit_plot = OrbitPlot(row_count=1, extra_col_count=0)
            qp = qp_0
            smo = smo_0

        orbit_plot.plot_curve(curve_description='initial', axis_v=orbit_plot.axis_vv[0], smo=smo_0)

        orbit_plot.plot_and_clear(
            filename=os.path.join(
                'shooting_method_3.custom_plot',
                construct_filename(
                    obj=smo.objective(),
                    t_delta=options.dt,
                    t_max=options.max_time,
                    initial_condition=qp,
                    t_min=smo.t_min()
                )
            )
        )
