import cmath
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import sys

sys.path.append('/Users/vdods/files/github/heisenberg/library')

import fourier

def vector (X):
    return np.array(X)

def matrix (X):
    row_count = len(X)
    assert row_count > 0
    column_count = len(X[0])
    assert all(len(row) == column_count for row in X)
    return np.array(X).reshape(row_count, column_count)

# def colvec (X):
#     a = np.array(X)
#     #return a.reshape((len(a.flat),1))
#     return a.reshape(len(a.flat))

# def rowvec (X):
#     a = np.array(X)
#     return a.reshape((1,len(a.flat)))

def squared_norm (X):
    return sum(x**2 for x in X.flat)

def squared_L2_norm (X):
    return sum(x**2 for x in X.flat) / len(X.flat)

def squared_complex_norm (z):
    return z.real**2 + z.imag**2

def squared_complex_L2_norm (Z):
    return sum(squared_complex_norm(z) for z in Z) / len(Z)

# def generated_zeta (M):
#     """
#     Returns a function which takes C^M and produces C^L, where L = (M+M)\{0}.
#     Here, M should be a container of integer-valued modes.
#     """
#     M_inv = {m:i for (i,m) in enumerate(M)}
#     L = frozenset(m1-m2 for m1 in M for m2 in M)
#     def zeta (C):
#         assert len(C) == len(M), 'not enough input params.'
#         return vector([sum((l+m)*C[M_inv[m]].conjugate()*C[M_inv[l+m]]/(2*l) if l != 0 else 0.0 for m in M if l+m in M) for l in L])
#     return zeta, L

def other_generated_zeta (M):
    M_inv = {m:i for i,m in enumerate(M)}
    L = list(frozenset(m1-m2 for m1 in M for m2 in M))
    L.sort()
    L_inv = {l:i for i,l in enumerate(L)}
    # T is the 3-tensor defined by T:(w \otimes z) = \bar{w}z, where w and z are
    # complex numbers identified as points in \mathbb{R}^2.
    T = np.zeros((2,2,2))
    T[0,0,0] =  1.0
    T[0,1,1] =  1.0
    T[1,0,1] =  1.0
    T[1,1,0] = -1.0
    # Mult is the 3-tensor defining the quadratic function zeta_M.
    Mult = np.zeros((2*len(L), 2*len(M), 2*len(M)))
    for l in L:
        if l == 0:
            continue
        i = L_inv[l]
        for m in M:
            if l+m not in M:
                continue
            j = M_inv[m]
            k = M_inv[l+m]
            Mult[2*i:2*(i+1),2*j:2*(j+1),2*k:2*(k+1)] += T*(l+m)/(2*l)
    
    def zeta (R):
        assert len(R.flat) == 2*len(M), 'not enough input params.'
        return np.einsum('ijk,j,k', Mult, R.flat, R.flat)
    
    def D_zeta ((R,V)):
        assert len(R.flat) == 2*len(M), 'not enough input params.'
        assert len(V.flat) == 2*len(M), 'not enough input params.'
        # Product rule (could probably simplify this due to some symmetry in Mult).
        return (zeta(R), np.einsum('ijk,j,k', Mult, V.flat, R.flat) + np.einsum('ijk,j,k', Mult, R.flat, V.flat))
    
    def D_zeta_total (R):
        assert len(R.flat) == 2*len(M), 'not enough input params.'
        # Product rule (could probably simplify this due to some symmetry in Mult).
        return (zeta(R), np.einsum('ijk,k', Mult, R.flat) + np.einsum('ijk,j', Mult, R.flat))
    
    return zeta,D_zeta,L

def as_real_vector (complex_vector):
    elements = []
    for z in complex_vector:
        elements.append(z.real)
        elements.append(z.imag)
    return vector(elements)

def test_other_generated_zeta ():
    M = [-3,-1,0,2,3,5,10]
    for _ in range(1000):
        C = vector([complex(np.random.randn(), np.random.randn()) for m in M])
        V_C = as_real_vector(C)
        zeta_0,L_0 = generated_zeta(M)
        zeta_1,D_zeta_1,L_1 = other_generated_zeta(M)
        delta = as_real_vector(zeta_0(C)) - zeta_1(V_C)
        assert squared_L2_norm(delta) < 1.0e-10
    print 'Test passed.'
    
#test_other_generated_zeta()

# def generated_Q (M, omega):
#     """
#     Returns a function which takes C^M and produces the coefficient of the linear portion of the (w,z) curve.
#     """
#     half_omega = 0.5*omega
#     def Q (C):
#         assert len(C) == len(M), 'not enough input params.'
#         return half_omega*sum(m*squared_complex_norm(c) for (m,c) in itertools.izip(M,C))
#     return Q

def other_generated_Q (M, omega):
    half_omega = 0.5*omega
    # TODO: get rid of M_inv here
    M_inv = {m:i for i,m in enumerate(M)}
    def Q (R):
        assert len(R.shape) == 1
        assert len(R) == 2*len(M), 'not enough input params.'
        return half_omega*sum(m*squared_norm(R[2*M_inv[m]:2*(M_inv[m]+1)]) for m in M)
    def D_Q ((R,V)):
        assert len(R.shape) == 1
        assert len(V.shape) == 1
        assert len(R) == 2*len(M), 'not enough input params.'
        assert len(V) == 2*len(M), 'not enough input params.'
        return (Q(R), omega*sum(m*np.einsum('i,i', R[2*M_inv[m]:2*(M_inv[m]+1)], V[2*M_inv[m]:2*(M_inv[m]+1)]) for m in M))
    def D_Q_total (R):
        assert len(R.shape) == 1
        assert len(R) == 2*len(M), 'not enough input params.'
        differential = np.ndarray((len(R)), dtype=complex)
        for m in M:
            i = M_inv[m]
            differential[2*i:2*(i+1)] = omega*m*R[2*i:2*(i+1)]
        return (Q(R), differential)
    return Q, D_Q

def test_other_generated_Q ():
    M = [-3,-1,0,2,3,5,10]
    for _ in range(1000):
        C = vector([complex(np.random.randn(), np.random.randn()) for m in M])
        V_C = as_real_vector(C)
        Q_0 = generated_Q(M, 1.0)
        Q_1,D_Q_1 = other_generated_Q(M, 1.0)
        delta = Q_0(C) - Q_1(V_C)
        assert squared_complex_norm(delta) < 1.0e-10
    print 'Test passed.'

#test_other_generated_Q()

def antiderivative_of (X, sample_times):
    s = 0.0
    retval = [s]
    deltas = [sample_times[i+1]-sample_times[i] for i in range(len(sample_times)-1)]
    for (x,delta) in itertools.izip(X,deltas):
        s += x*delta
        retval.append(s)
    return vector(retval)

def derivative_of (X, sample_times):
    deltas = [sample_times[i+1]-sample_times[i] for i in range(len(sample_times)-1)]
    return vector([(X[i+1]-X[i])/delta for i,delta in enumerate(deltas)])

def definite_integral (sample_times):
    assert sample_times[-1] - sample_times[0] > 0, 'Must be positively oriented, nonempty sampled interval.'
    L = sample_times[-1] - sample_times[0]
    integral_covector = np.ndarray((len(sample_times)-1,), dtype=float)
    for i in  range(len(sample_times)-1):
        integral_covector[i] = sample_times[i+1]-sample_times[i]
    def I (X):
        return np.einsum('j,j', integral_covector, X)
    def D_I ((X,V)):
        return (I(X),I(V))
    def D_I_total (X):
        return (I(X), integral_covector)
    return I, D_I

def imag (z):
    return z.imag

def D_imag ((z,v)):
    return (imag(z),imag(v))

#def D_imag_total (z):
#    return (imag(z), )

def imag_v (Z):
    #print 'in imag_v, len(Z) = {0}, type(Z[0]) = {1}, Z[0].shape = {2}'.format(len(Z), type(Z[0]), Z[0].shape)
#     retval = colvec([imag(z) for z in Z]).reshape(len(Z))
    retval = vector([imag(z) for z in Z]) #np.array([imag(z) for z in Z])
    #print 'in imag_v, len(Z) = {0}, len(retval) = {1}'.format(len(Z), len(retval))
    return retval

def D_imag_v ((Z,V)):
    #print 'in D_imag_v, len(Z) = {0}, len(V) = {1}'.format(len(Z), len(V))
    return (imag_v(Z),imag_v(V))

def realify (z):
    return (z.real, z.imag)

def D_realify ((z,v)):
    return (realify(z),realify(v))

def realify_v (Z):
    #retval = colvec([[z.real, z.imag] for z in Z])
    #retval = np.array([[z.real, z.imag] for z in Z]).reshape(2*len(Z))
    retval = matrix([[z.real, z.imag] for z in Z]).reshape(2*len(Z)) # This flattens the matrix, in row-major order.
    #print 'in realify_v, len(Z) = {0}, len(retval) = {1}, type(retval[0]) = {2}'.format(len(Z), len(retval), type(retval[0]))
    return retval

def D_realify_v ((Z,V)):
    #print 'in D_realify_v, len(Z) = {0}, len(V) = {1}'.format(len(Z), len(V))
    return (realify_v(Z),realify_v(V))

def complexify (x):
    assert len(x) == 2
    return complex(x[0], x[1])

def D_complexify ((x,v)):
    return (complexify(x),complexify(v))

def complexify_v (X):
    assert len(X.flat) % 2 == 0
    n = len(X.flat) / 2
    return vector([complex(X.flat[2*i], X.flat[2*i+1]) for i in range(n)])

def D_complexify_v ((X,V)):
    return (complexify_v(X),complexify_v(V))

def chi (U):
    assert len(U) == 2
    #print 'in chi(U), len(U[0]) = {0}, len(U[1]) = {1}'.format(len(U[0]), len(U[1]))
    assert len(U[0]) == 2*len(U[1])
    n = len(U[1])
    #print 'in chi(U), type(U[0]) = {0}, type(U[1]) = {1}, U[0] = {2}, U[1] = {3}'.format(type(U[0]), type(U[1]), U[0], U[1])
    #print 'buffer = {0}'.format([[x,y,z] for ((x,y),z) in itertools.izip(U[0],U[1])])
    #return np.ndarray((n,3), dtype=float, buffer=np.array([[U[0][2*i],U[0][2*i+1],U[1][i]] for i in range(n)])).T
    return matrix([[U[0][2*i],U[0][2*i+1],U[1][i]] for i in range(n)])

def test_chi ():
    Theta = np.linspace(0.0, 1.0, 11)
    A = [(x,x**2) for x in Theta]
    B = [2.0*x for x in Theta]
    C = chi([A,B])
    print C

def D_chi ((U,V)):
    # chi is a linear map, so its differential is very simple.
    #print 'in D_chi, type(U[0]) = {0}, type(U[1]) = {1}, type(V[0]) = {2}, type(V[1]) = {3}'.format(type(U[0]), type(U[1]), type(V[0]), type(V[1]))
    #print 'U[0].shape = {0}, U[1].shape = {0}, V[0].shape = {0}, V[1].shape = {0}'.format(U[0].shape, U[1].shape, V[0].shape, V[1].shape)
    #print 'U = {0}'.format(U)
    assert len(U) == 2
    assert len(V) == 2
    X = chi((U[0],V[0]))
    Y = chi((U[1],V[1]))
    return (X,Y)
                
def eta (U):
    assert len(U) == 2
    assert U[0].shape == U[1].shape
    n = U[0].shape[1]
    retval = np.ndarray((U[0].shape[0],2*n), dtype=float)
    retval[:,:n] = U[0]
    retval[:,n:] = U[1]
    return retval

def test_eta ():
    U = [np.random.randn(*(3,5)) for _ in range(2)]
    eta_U = eta(U)
    print 'U[0] = {0}'.format(U[0])
    print 'U[1] = {0}'.format(U[1])
    print 'eta_U = {0}'.format(eta_U)

def D_eta ((U,V)):
    # eta is a linear map, so its differential is very simple.
    #return (eta(U),eta(V))
    return (eta((U[0],V[0])),eta((U[1],V[1])))

def cartesian_product_of_functions_with_shared_domain (*functions):
    def _ (x):
        retval = tuple(f(x) for f in functions)
        #print 'in cartesian_product_of_functions_with_shared_domain, retval types = {0}'.format(tuple((type(t),len(t),len(t[0]),len(t[1])) for t in retval))
        return retval
    return _

def composition_of_functions (*functions):
    def _ (x):
        retval = x
        for f in reversed(functions):
            retval = f(retval)
        return retval
    return _

def sum_of_functions_with_shared_domain (*functions):
    def _ (x):
        #print 'in retval of sum_of_functions_with_shared_domain, {0}'.format([type(f(x)) for f in functions])
        return sum(f(x) for f in functions)
    return _

def sum_of_tuples (*tuples):
    return (sum(t) for t in itertools.izip(*tuples))

def direct_sum_of_functions_with_shared_domain (*functions):
    return lambda x : sum_of_tuples(*[f(x) for f in functions])

alpha = 2.0 / math.pi # 1.0
beta = 16.0 # This used to incorrectly be 1/16

def Lagrangian (pv):
    mu = (pv[0]**2 + pv[1]**2)**2 + beta*pv[2]**2
    # First term is kinetic energy, second is negative potential.
    return 0.5*(pv[3]**2 + pv[4]**2) + alpha/math.sqrt(mu)

def D_Lagrangian ((pv, pv_prime)):
    assert len(pv) == 6
    assert len(pv_prime) == 6
    r_squared = pv[0]**2 + pv[1]**2
    mu = (r_squared)**2 + beta*pv[2]**2
    mu_to_negative_three_halves = pow(mu,-1.5)
    # TODO: factor this so it's fewer operations
    return (Lagrangian(pv), \
              -2.0*alpha*r_squared*pv[0]*mu_to_negative_three_halves*pv_prime[0] \
            + -2.0*alpha*r_squared*pv[1]*mu_to_negative_three_halves*pv_prime[1] \
            + -alpha*beta*pv[2]*mu_to_negative_three_halves*pv_prime[2] \
            + pv[3]*pv_prime[3] \
            + pv[4]*pv_prime[4])

def Lagrangian_v (PV):
    #retval = np.ndarray((1,PV.shape[1]), dtype=float)
    #for c in range(PV.shape[1]):
    #    retval[0,c] = Lagrangian(PV[:,c])
    #return retval
    return vector([Lagrangian(pv) for pv in PV])

def D_Lagrangian_v ((PV,PV_prime)):
    # Each of PV and PV_prime are matrices of size 6 by N, where N is the number of samples
    assert PV.shape == PV_prime.shape
    assert PV.shape[1] == 6
    retval = (np.ndarray((PV.shape[0],), dtype=float), np.ndarray((PV.shape[0],), dtype=float))
    for r,pv_pv_prime in enumerate(itertools.izip(PV,PV_prime)):
        DL = D_Lagrangian(pv_pv_prime)
        retval[0][r] = DL[0]
        retval[1][r] = DL[1]
    return retval

def Hamiltonian (pv):
    mu = (pv[0]**2 + pv[1]**2)**2 + beta*pv[2]**2
    # First term is kinetic energy, second is potential.
    return 0.5*(pv[3]**2 + pv[4]**2) - alpha/math.sqrt(mu)

def Hamiltonian_v (PV):
    #retval = np.ndarray((1,PV.shape[1]), dtype=float)
    #for c in range(PV.shape[1]):
    #    retval[0,c] = Hamiltonian(PV[:,c])
    #return retval
    return vector([Hamiltonian(pv) for pv in PV])

def cotangent_vector (tangent_vector):
    assert len(tangent_vector) == 6
    #   x' = p_x - 0.5*y*p_z
    #   y' = p_y + 0.5*x*p_z
    #   z' = p_z
    # This implies that
    #   p_x = x' + 0.5*y*z'
    #   p_y = y' - 0.5*x*z'
    #   p_z = z'
    x = tangent_vector[0]
    y = tangent_vector[1]
    z = tangent_vector[2]
    x_dot = tangent_vector[3]
    y_dot = tangent_vector[4]
    z_dot = tangent_vector[5]
    return vector([x, y, z, x_dot+0.5*y*z_dot, y_dot-0.5*x*z_dot, z_dot])

#test_chi()
#test_eta()

def main ():
    period = 273.5
    omega = cmath.exp(2.0j*math.pi/period)
    #M = frozenset([-8,-5,-2,1,4,7,10])
    k = 3
    fold = 3
    M = range(1-fold*k,1+fold*k+1,fold)
    M_inv = {m:i for i,m in enumerate(M)}
    #zeta_M,L = generated_zeta(M)
    zeta_M,D_zeta_M,L = other_generated_zeta(M)
    L_inv = {l:i for i,l in enumerate(L)}
    #Q_M = generated_Q(M, omega)
    Q_M, D_Q_M = other_generated_Q(M, omega)
    
    sample_count = 1000
    sample_times = np.linspace(0.0, period, sample_count)
    
    F_M = fourier.Transform(M, sample_times)
    F_L = fourier.Transform(L, sample_times)
    I, D_I = definite_integral(sample_times)
    sample_times = sample_times[:-1]

    linear = sample_times
    #linear_term = lambda R : linear*Q_M(R)
    def linear_term (R):
        Q = Q_M(R)
        return vector([Q*t for t in sample_times])
    def D_linear_term ((R,V)):
        d = D_Q_M((R,V))
        return (linear*d[0], linear*d[1])
    
    constant = vector([1.0 for _ in sample_times])
    def constant_term (R):
        Q = Q_M(R)
        return vector([Q for _ in sample_times])
    def D_constant_term ((R,V)):
        d = D_Q_M((R,V))
        return (constant*d[0], constant*d[1])
    
    # Fourier transforms
    #FT_M = lambda C : F_M.sampled_sum_of(C)
    #FT_L = lambda C : F_L.sampled_sum_of(C)
    def FT_M (C):
        retval = F_M.sampled_sum_of(C)
        return retval.reshape(retval.size)
    def FT_L (C):
        retval = F_L.sampled_sum_of(C)
        return retval.reshape(retval.size)
    
    # FT_M and FT_L are linear, so their differentials are very simple.
    D_FT_M = lambda (C,V) : (FT_M(C),FT_M(V))
    D_FT_L = lambda (C,V) : (FT_L(C),FT_L(V))
                
    # Derivative with respect to Fourier coefficients
    #FCD_M = lambda C : F_M.coefficients_of_derivative_of(C)
    #FCD_L = lambda C : F_L.coefficients_of_derivative_of(C)
    def FCD_M (C):
        retval = F_M.coefficients_of_derivative_of(C)
        return retval.reshape(retval.size)
    def FCD_L (C):
        retval = F_L.coefficients_of_derivative_of(C)
        return retval.reshape(retval.size)
    
    # FCD_M and FCD_L are linear, so their differentials are very simple.
    D_FCD_M = lambda (C,V) : (FCD_M(C),FCD_M(V))
    D_FCD_L = lambda (C,V) : (FCD_L(C),FCD_L(V))
    
    position = \
        composition_of_functions( \
            chi, \
            cartesian_product_of_functions_with_shared_domain( \
                composition_of_functions(realify_v, FT_M, complexify_v), \
                composition_of_functions( \
                    imag_v, \
                    sum_of_functions_with_shared_domain( \
                        composition_of_functions(FT_L, complexify_v, zeta_M), \
                        linear_term \
                    ) \
                ) \
            ) \
        )
    velocity = \
        composition_of_functions( \
            chi, \
            cartesian_product_of_functions_with_shared_domain( \
                composition_of_functions(realify_v, FT_M, FCD_M, complexify_v), \
                composition_of_functions( \
                    imag_v, \
                    sum_of_functions_with_shared_domain( \
                        composition_of_functions(FT_L, FCD_L, complexify_v, zeta_M), \
                        constant_term \
                    ) \
                ) \
            ) \
        )
    
    use_Q_contribution = False
    if not use_Q_contribution:
        action = \
            composition_of_functions( \
                I, \
                Lagrangian_v, \
                eta, \
                cartesian_product_of_functions_with_shared_domain(position, velocity) \
            )
    else:
        action = \
            sum_of_functions_with_shared_domain( \
                composition_of_functions( \
                    I, \
                    Lagrangian_v, \
                    eta, \
                    cartesian_product_of_functions_with_shared_domain(position, velocity) \
                ), \
                composition_of_functions( \
                    lambda x : 100000.0*x**2, \
                    imag, \
                    Q_M
                ) \
            )
    
    D_position = \
        composition_of_functions( \
            D_chi, \
            cartesian_product_of_functions_with_shared_domain( \
                composition_of_functions(D_realify_v, D_FT_M, D_complexify_v), \
                composition_of_functions( \
                    D_imag_v, \
                    direct_sum_of_functions_with_shared_domain( \
                        composition_of_functions(D_FT_L, D_complexify_v, D_zeta_M), \
                        D_linear_term \
                    ) \
                ) \
            ) \
        )
    D_velocity = \
        composition_of_functions( \
            D_chi, \
            cartesian_product_of_functions_with_shared_domain( \
                composition_of_functions(D_realify_v, D_FT_M, D_FCD_M, D_complexify_v), \
                composition_of_functions( \
                    D_imag_v, \
                    direct_sum_of_functions_with_shared_domain( \
                        composition_of_functions(D_FT_L, D_FCD_L, D_complexify_v, D_zeta_M), \
                        D_constant_term \
                    ) \
                ) \
            ) \
        )
    D_action = \
        composition_of_functions( \
            D_I, \
            D_Lagrangian_v, \
            D_eta, \
            cartesian_product_of_functions_with_shared_domain(D_position, D_velocity) \
        )
    
    constraint = composition_of_functions(imag, Q_M)
    D_constraint = composition_of_functions(D_imag, D_Q_M)

    def standard_basis_vector (dim, index):
        return vector([1.0 if i == index else 0.0 for i in range(dim)])
    
    def D_action_total (R):
        # This is super wasteful, but for now it's ok
        dim = len(R)
        return vector([D_action((R,standard_basis_vector(dim,i)))[1] for i in range(dim)])
    
    def D_constraint_total (R):
        # This is super wasteful, but for now it's ok
        dim = len(R)
        return vector([D_constraint((R,standard_basis_vector(dim,i)))[1] for i in range(dim)])
    
    def Lambda (R_lagmult):
        R = R_lagmult[:-1]
        lagmult = R_lagmult[-1]
        return action(R) + lagmult*constraint(R)
    
    def D_Lambda_total (R_lagmult):
        R = R_lagmult[:-1]
        lagmult = R_lagmult[-1]
        dLambda_dR = D_action_total(R) + lagmult*D_constraint_total(R)
        dLambda_dlagmult = constraint(R)
        retval = np.ndarray((len(R_lagmult),), dtype=float)
        retval[:-1] = dLambda_dR
        retval[-1] = dLambda_dlagmult
        return retval

    def objective_function (R_lagmult):
        return squared_L2_norm(D_Lambda_total(R_lagmult))
    
    #TODO: blah... actually need the total differential, not an evaluated differential.

#     #R = realify_v(colvec([complex(np.random.randn(), np.random.randn())/(m if m != 0 else 1) for m in M]))
#     R = np.random.randn(2*len(M))
#     for i,m in enumerate(M):
#         #i = M_inv[m]
#         if m != 0:
#             R[2*i:2*(i+1)] /= abs(m)
#     #C = colvec([complex(np.random.randn(), np.random.randn()) for m in M])

    R_lagmult = np.random.randn(2*len(M)+1) # The last component is lagmult.
    print 'len(R_lagmult) = {0}'.format(len(R_lagmult))
    # Optimize with respect to objective_function.  This defines the constrained dynamics problem.
    #R_lagmult = scipy.optimize.fmin_powell(objective_function, R_lagmult, disp=True)
    
    R = R_lagmult[:-1]
    lagmult = R_lagmult[-1]
    
    print 'Lambda(R_lagmult) = {0}'.format(Lambda(R_lagmult))

    # First find an initial condition which has constraint(R) near 0.
    R = scipy.optimize.fmin_powell(lambda R : constraint(R)**2, R, disp=True)
    # Attempt to optimize to find the argmin of the action functional.
    #R = scipy.optimize.fmin_powell(action, R, ftol=1.0e-5, disp=True)

    A = action(R)
    print 'action = {0}'.format(A)
    print 'constraint = {0}'.format(constraint(R))
    print 'D_position(R)*e0 squared L2 norm = {0}'.format(squared_L2_norm(D_position((R,standard_basis_vector(len(R),0)))[1]))
    print 'D_velocity(R)*e0 squared L2 norm = {0}'.format(squared_L2_norm(D_velocity((R,standard_basis_vector(len(R),0)))[1]))
    print 'D_action(R)*e0 = {0}'.format(D_action((R,standard_basis_vector(len(R),0))))
    print 'D_constraint(R)*e0 = {0}'.format(D_constraint((R,standard_basis_vector(len(R),0))))
    D_action_total_R = D_action_total(R)
    print 'D_action_total(R) = {0}'.format(D_action_total_R)
    print 'D_action_total(R) squared norm = {0}'.format(squared_L2_norm(D_action_total_R))
    D_constraint_total_R = D_constraint_total(R)
    print 'D_constraint_total(R) = {0}'.format(D_constraint_total_R)
    print 'D_constraint_total(R) squared norm = {0}'.format(squared_L2_norm(D_constraint_total_R))
    
    print 'alpha = {0}, beta = {1}'.format(alpha, beta)
    print 'R = {0}'.format(R)
    print 'M = {0}'.format(M)
    print 'Q_M(R) = {0}'.format(Q_M(R))
    print 'period = {0}'.format(period)
    print 'sample_count = {0}'.format(sample_count)

    P = position(R)
    print 'P.shape = {0}'.format(P.shape)
    dP = derivative_of(P, sample_times)
    print 'dP.shape = {0}'.format(dP.shape)
    V = velocity(R)
    print 'V.shape = {0}'.format(V.shape)

    # Sanity check that the discrete derivative dP is about equal to V.
    print 'diagram commutation failure amount (should be near zero): {0}'.format(squared_L2_norm(dP - V[:-1,:]))
    
    plt.figure(1, figsize=(10,15))
    plt.subplot(3,2,1)
    plt.title('image of (x(t),y(t))')
    #plt.axes().set_aspect('equal')
    plt.plot(P[:,0], P[:,1])
    
    #plt.figure(3)
    plt.subplot(3,2,2)
    plt.title('z(t)')
    plt.plot(sample_times, P[:,2])
    
    plt.subplot(3,2,3)
    plt.title('image of (x\'(t),y\'(t))')
    #plt.axes().set_aspect('equal')
    plt.plot(V[:,0], V[:,1])
    
    #plt.figure(4)
    plt.subplot(3,2,4)
    plt.title('z\'(t)')
    plt.plot(sample_times, V[:,2])

    PV = eta([P, V])
    H_of_PV = Hamiltonian_v(PV)
    L_of_PV = Lagrangian_v(PV)
    
    print 'len(sample_times) = {0}, len(H_of_PV) = {1}, len(L_of_PV) = {2}'.format(len(sample_times), len(H_of_PV), len(L_of_PV))
    
    plt.subplot(3,2,5)
    plt.title('H(t)')
    plt.plot(sample_times, H_of_PV)
    
    #plt.figure(6)
    plt.subplot(3,2,6)
    plt.title('L(t)')
    plt.plot(sample_times, L_of_PV)

    plt.savefig('ostrich.png')
        
    #return R,M,period,sample_count

if __name__ == "__main__":
    main()
