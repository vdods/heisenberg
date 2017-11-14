import abc
from . import hamiltonian_dynamics_context
import numpy as np
import sympy as sp
import vorpy.symbolic

class Base(hamiltonian_dynamics_context.HamiltonianDynamicsContext):
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
        alpha     = 1/(8*cls.pi())
        beta      = 16
        r_squared = qp[0,0]**2 + qp[0,1]**2
        mu        = r_squared**2 + beta*qp[0,2]**2
        return -alpha / cls.sqrt(mu)

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
        alpha                              = 1/(8*cls.pi())
        beta                               = 16
        P_x                                = p[0] - q[1]*p[2]/2
        P_y                                = p[1] + q[0]*p[2]/2
        r_squared                          = q[0]**2 + q[1]**2
        mu                                 = r_squared**2 + beta*q[2]**2
        alpha_times_mu_to_neg_three_halves = alpha / cls.sqrt(mu)**3
        return np.array((
             P_y*p[2]/2 + alpha_times_mu_to_neg_three_halves*r_squared*2*q[0],
            -P_x*p[2]/2 + alpha_times_mu_to_neg_three_halves*r_squared*2*q[1],
             beta*alpha_times_mu_to_neg_three_halves*q[2]
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

class Symbolic(Base):
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

    @staticmethod
    def valid_embedding_dimensions ():
        return [1,2,3,5]

    @staticmethod
    def assert_is_valid_embedding_dimension (N):
        assert N in Symbolic.valid_embedding_dimensions(), 'invalid N (which is {0}); must be one of {1}.'.format(N, Symbolic.valid_embedding_dimensions())

    @classmethod
    def embedding_solver (cls, *, N, sheet_index):
        """
        With qp denoting the (2,3)-shaped symbolic coordinates

            [x  , y  , z  ]
            [p_x, p_y, p_z],

        this function symbolically solves

            H(qp) = 0

        for p_z, where particular submanifolds of the full 5-dimensional parameter space are used
        for different values of the N-dimensional embedded parameter space.  N must be in [1,2,3,5].

        If N = 1, then                       If N = 3, then

            x   = 1,                             x is free,
            y   = 0,                             y   = 0
            z   = 0,                             z   = 0,
            p_x = 0,                             p_x is free,
            p_y is free,                         p_y is free,
            p_z is solved for.                   p_z is solved for.

        If N = 2, then                       If N = 5, then

            x   = 1,                             x is free,
            y   = 0,                             y is free,
            z   = 0,                             z is free,
            p_x is free,                         p_x is free,
            p_y is free,                         p_y is free,
            p_z is solved for.                   p_z is solved for.

        The N = 5 case is the full parameterization of [one sheet of] the H = 0 submanifold.
        """
        Symbolic.assert_is_valid_embedding_dimension(N)
        assert 0 <= sheet_index < 2

        zero = sp.Integer(0)
        one = sp.Integer(1)

        x,y,z,p_x,p_y,p_z = sp.symbols(('x','y','z','p_x','p_y','p_z'))

        # The embedding is different depending on the dimension.
        if N == 1:
            qp = np.array((
                ( one,zero,zero),
                (zero, p_y, p_z),
            ))
            slice_coordinates = np.array((qp[1,1], qp[1,2])) # This is (p_y,p_z)
        elif N == 2:
            qp = np.array((
                (one,zero,zero),
                (p_x, p_y, p_z),
            ))
            slice_coordinates = np.array((qp[1,0], qp[1,1], qp[1,2])) # This is (p_x,p_y,p_z)
        elif N == 3:
            qp = np.array((
                (x  ,zero,zero),
                (p_x, p_y, p_z),
            ))
            slice_coordinates = np.array((qp[0,0], qp[1,0], qp[1,1], qp[1,2])) # This is (x,p_x,p_y,p_z)
        elif N == 5:
            qp = np.array((
                (x  , y  , z  ),
                (p_x, p_y, p_z),
            ))
            slice_coordinates = np.array((qp[0,0], qp[0,1], qp[0,2], qp[1,0], qp[1,1], qp[1,2])) # This is (x,y,z,p_x,p_y,p_z)

        #print('qp:')
        #print(qp)
        #print('slice_coordinates: {0}'.format(slice_coordinates))
        assert slice_coordinates.shape == (N+1,)
        embedding_domain = slice_coordinates[:N]
        #print('embedding_domain: {0}'.format(embedding_domain))
        assert embedding_domain.shape == (N,)
        assert slice_coordinates[-1] == p_z

        H = cls.H(qp)
        #print('H(qp) = {0}'.format(H))
        p_z_solution_v = sp.solve(H, p_z)
        #print('There are {0} solutions for the equation: {1} = 0'.format(len(p_z_solution_v), H))
        #for i,p_z_solution in enumerate(p_z_solution_v):
            #print('    solution {0}: p_z = {1}'.format(i, p_z_solution))
        # Take the solution specified by sheet_index
        p_z_solution = p_z_solution_v[sheet_index]

        # Create the embedding, which maps embedding_domain |-> embedding,
        # where in particular, the p_z coordinate has been replaced by its solution.
        embedding = np.copy(qp)
        embedding[1,2] = p_z_solution

        return embedding_domain,embedding

class Numeric(Base):
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

    @staticmethod
    def embedding (*, N, sheet_index):
        Symbolic.assert_is_valid_embedding_dimension(N)

        def symbolic_embedding_function_creator ():
            embedding_domain,embedding  = Symbolic.embedding_solver(N=N, sheet_index=sheet_index)
            replacement_d               = {
                'array'         :'np.array',
                'ndarray'       :'np.ndarray',
                'dtype=object'  :'dtype=float',
                'sqrt'          :'np.sqrt',
                'pi'            :'np.pi',
            }
            argument_id                 = 'X' # This is arbitrary, but should just avoid conflicting with any of the replacements.
            import_v                    = ['import numpy as np']
            decorator_v                 = []

            return embedding, embedding_domain, replacement_d, argument_id, import_v, decorator_v

        return vorpy.symbolic.cached_lambdified(
            'heisenberg_dynamics_context__embedding_{0}_{1}'.format(N, sheet_index),
            function_creator=symbolic_embedding_function_creator,
            verbose=False
        )
