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
        return 2/cls.pi()
        #return cls.pi()/8

    @classmethod
    def beta (cls):
        """Return the beta value occurring in the fundamental solution to the sub-Riemannian Laplacian."""
        return 16

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

    def __solve_for_embedding2 (self):
        # Symbolically solve H(qp) = 0 for qp[1,2].
        X = np.array(sp.symbols(('p_x','p_y','p_z')))
        zero = sp.Integer(0)
        one = sp.Integer(1)
        qp = np.array(
            (
                ( one, zero, zero),
                (X[0], X[1], X[2]),
            ),
            dtype=object
        )
        H = Symbolic.H(qp)
        #print('H(qp) = {0}'.format(H))
        p_z = qp[1,2] # Momentum for z coordinate
        p_z_solution_v = sp.solve(H, p_z)
        #print('len(p_z_solution_v) = {0}'.format(len(p_z_solution_v)))
        #print('p_z_solution_v = {0}'.format(p_z_solution_v))
        # Just take the last solution.
        p_z_solution = p_z_solution_v[-1]
        #print('p_z_solution = {0}'.format(p_z_solution))
        # The domain for this function is
        #     -sqrt(4/pi) <= p_x <= sqrt(4/pi)

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
        X = np.array(sp.symbols(('x','p_x','p_y','p_z')))
        zero = sp.Integer(0)
        qp = np.array(
            (
                (X[0], zero, zero),
                (X[1], X[2], X[3]),
            ),
            dtype=object
        )
        H = Symbolic.H(qp)
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

    def __solve_for_embedding5 (self):
        # Symbolically solve H(qp) = 0 for qp[1,2].
        X = np.array(sp.symbols(('x','y','z','p_x','p_y','p_z')))
        zero = sp.Integer(0)
        qp = np.array(
            (
                (X[0], X[1], X[2]),
                (X[3], X[4], X[5]),
            ),
            dtype=object
        )
        H = Symbolic.H(qp)
        #print('H(qp) = {0}'.format(H))
        p_z = qp[1,2] # Momentum for z coordinate
        p_z_solution_v = sp.solve(H, p_z)
        #print('len(p_z_solution_v) = {0}'.format(len(p_z_solution_v)))
        #print('p_z_solution_v = {0}'.format(p_z_solution_v))
        # Just take the last solution.
        p_z_solution = p_z_solution_v[-1]
        #print('p_z_solution = {0}'.format(p_z_solution))

        self.symbolic_embedding5_domain = X[:5]
        self.symbolic_embedding5 = np.array(
            (
                (X[0], X[1],         X[2]),
                (X[3], X[4], p_z_solution),
            ),
            dtype=object
        )
        self.embedding5 = vorpy.symbolic.lambdified(
            self.symbolic_embedding5,
            self.symbolic_embedding5_domain,
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
        self.__solve_for_embedding5()

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
        #H = Symbolic.H(qp)
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
        #H = Symbolic.H(qp)
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

