import sys
sys.path.append('library')

import fractions
import math
import numpy as np
import realified_fourier
import symbolic
import tensor
import time

def call_func_and_print_timing_info (func_name, func, *args, **kwargs):
    print 'calling {0} ...'.format(func_name)
    start = time.time()
    retval = func(*args, **kwargs)
    print '{0} took {1} s.'.format(func_name, time.time() - start)
    return retval

# This and einsum_for_two is from http://stackoverflow.com/questions/15606937/how-do-i-get-numpy-einsum-to-play-well-with-sympy
def alt_einsum(string, *args):
    index_groups = map(list, string.split(','))
    assert len(index_groups) == len(args)
    tensor_indices_tuples = zip(index_groups, args)
    return reduce(einsum_for_two, tensor_indices_tuples)[1]

def einsum_for_two(tensor_indices1, tensor_indices2):
    string1, tensor1 = tensor_indices1
    string2, tensor2 = tensor_indices2
    sum_over_indices = set(string1).intersection(set(string2))
    new_string = string1 + string2
    axes = ([], [])
    for i in sum_over_indices:
        new_string.remove(i)
        new_string.remove(i)
        axes[0].append(string1.index(i))
        axes[1].append(string2.index(i))
    return new_string, np.tensordot(tensor1, tensor2, axes)

def generate_zeta (M, contraction):
    # TODO: get rid of M_inv if possible
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
    # zeta_tensor is the 3-tensor defining the quadratic function zeta_M.
    zeta_tensor = np.zeros((2*len(L), 2*len(M), 2*len(M)))
    for l in L:
        if l == 0:
            continue
        i = L_inv[l]
        for m in M:
            if l+m not in M:
                continue
            j = M_inv[m]
            k = M_inv[l+m]
            zeta_tensor[2*i:2*(i+1),2*j:2*(j+1),2*k:2*(k+1)] += T*(l+m)/(2*l)
    
    def zeta (R):
        assert len(R) == 2*len(M), 'not enough input params.'
        if contraction == 'np.einsum':
            return np.einsum('ijk,j,k', zeta_tensor, R, R)
        elif contraction == 'tensor.contract':
            return tensor.contract('ijk,j,k', zeta_tensor, R, R, dtype=R.dtype)
        else:
            return alt_einsum('ijk,j,k', zeta_tensor, R, R)
    
    return zeta,L,zeta_tensor

def generate_imag_Q (M, period):
    # This is the imaginary part of 0.5*cmath.exp(2.0j*math.pi/period).
    half_imag_rotation = 0.5*math.sin(2.0*math.pi/period)

    def imag_Q (R):
        assert len(R.shape) == 1
        assert len(R) == 2*len(M), 'not enough input params.'
        return half_imag_rotation*sum(m*(R[2*i]**2 + R[2*i+1]**2) for i,m in enumerate(M))

    return imag_Q

def generate_integrate_over_sample_times (sample_times, period, contraction):
    assert len(sample_times) > 0, 'sample_times must be nonempty.'
    assert all(sample_times[i+1] > sample_times[i] for i in range(len(sample_times)-1)), 'sample_times must be a strictly increasing sequence.'
    assert period > sample_times[-1], 'period must be greater than last element of sample_times.'
    L = period - sample_times[0]
    integral_covector = np.array([sample_times[i+1] - sample_times[i] for i in range(len(sample_times)-1)] + [period - sample_times[-1]])
    assert len(integral_covector) == len(sample_times)
    def integrate_over_sample_times (X):
        if contraction == 'np.einsum':
            return np.einsum('j,j', integral_covector, X)
        elif contraction == 'tensor.contract':
            return tensor.contract('j,j', integral_covector, X, dtype=X.dtype)
        else:
            return alt_einsum('j,j', integral_covector, X)
    return integrate_over_sample_times

def generate_imag_projection_over_sample_times (sample_times):
    imag_projection_matrix = np.ndarray((len(sample_times),2*len(sample_times)), \
                                        dtype=float, \
                                        buffer=np.array([[1.0 if (c%2==1 and c//2==r) else 0.0 for c in range(2*len(sample_times))] \
                                                                                               for r in range(len(sample_times))]))
    def imag_projection_over_sample_times (wz_samples):
        return imag_projection_matrix.dot(wz_samples)
    return imag_projection_over_sample_times,imag_projection_matrix

def chi ((R2_vector,R_vector)):
    """Interleave (R^2)^n with R^n to make (R^3)^n."""
    assert len(R2_vector) == 2*len(R_vector)
    n = len(R_vector)
    return np.array([[R2_vector[2*i],R2_vector[2*i+1],R_vector[i]] for i in range(n)])

def eta ((U,V)):
    assert len(U.shape) == 2
    assert U.shape[1] == 3
    assert U.shape == V.shape
    n = U.shape[1]
    retval = np.ndarray((U.shape[0],2*n), dtype=U[0].dtype)
    retval[:,:n] = U
    retval[:,n:] = V
    return retval

def generate_hamiltonian (alpha, beta):
    def hamiltonian (pv):
        assert len(pv) == 6
        mu = (pv[0]**2 + pv[1]**2)**2 + beta*pv[2]**2
        # First term is kinetic energy, second is potential.
        return 0.5*(pv[3]**2 + pv[4]**2) - alpha*mu**(-0.5)
    return hamiltonian

def generate_hamiltonian_v (hamiltonian):
    def hamiltonian_v (PV):
        assert PV.shape[1] == 6
        return np.array([hamiltonian(pv) for pv in PV])
    return hamiltonian_v

def generate_hamiltonian_vector_field (alpha, beta):
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
        mu = r**2 + beta*z**2
        # alpha = 2.0/math.pi
        # alpha = 1.0
        alpha_times_mu_to_neg_three_halves = alpha*mu**(-1.5)
        return np.array([P_x, \
                         P_y, \
                          0.5*x*P_y - 0.5*y*P_x, \
                         -0.5*P_y*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*x, \
                          0.5*P_x*p_z - alpha_times_mu_to_neg_three_halves*r*2.0*y, \
                         -16.0*alpha_times_mu_to_neg_three_halves*z],
                         dtype=float)
    return hamiltonian_vector_field

def generate_lagrangian (alpha, beta):
    def lagrangian (pv):
        # NOTE that this doesn't use pv[5] (i.e. dz/dt) at all.
        assert len(pv) == 6
        mu = (pv[0]**2 + pv[1]**2)**2 + beta*pv[2]**2
        # First term is kinetic energy, second is potential.
        return 0.5*(pv[3]**2 + pv[4]**2) + alpha*mu**(-0.5)
    return lagrangian

def generate_lagrangian_v (lagrangian):
    def lagrangian_v (PV):
        assert PV.shape[1] == 6
        return np.array([lagrangian(pv) for pv in PV])
    return lagrangian_v

def load_cache_or_compute (cache_filename, computation, *args, **kwargs):
    import pickle
    
    try:
        print 'attempting to unpickle from file \'{0}\'.'.format(cache_filename)
        cached_value = pickle.load(open(cache_filename, 'r'))
        print 'unpickling succeeded -- returning unpickled value.'
        return cached_value
    except:
        print 'unpickling failed -- computing value.'
        start = time.time()
        computed_value = computation(*args, **kwargs)
        print 'value computed in {0} s.'.format(time.time() - start)
        try:
            print 'attempting to pickle computed value to file \'{0}\'.'.format(cache_filename)
            pickle.dump(computed_value, open(cache_filename, 'w'))
            print 'pickling succeeded -- returning computed value.'
        except:
            print 'WARNING: Could not pickle data to file \'{0}\' -- returning computed value.'.format(cache_filename)
        return computed_value

class ProblemContext:
    def __init__ (self, **kwargs):
        """
        Required keyword arguments:
            symmetry_degree : Positive integer indicating the degree of symmetry (e.g. 3-fold, 5-fold).
            symmetry_class  : Positive integer, coprime to symmetry_degree, indicating the particular
                              class of symmetry.  TODO: describe this
            xy_mode_count   : The number of modes used in the Fourier sum for the xy curve.
            sample_count    : A positive integer indicating how many samples will be used to represent the xy curve.
            period          : A positive float indicating the period of the curve.
            contraction     : Specifies the method used to contract tensors.  One of: 'np.einsum', 'tensor.contract',
                              'alt_einsum'.  Note that dtype=object can't use np.einsum (for some dumb reason).
                              Default is 'np.einsum', because it is the fastest.
            alpha           : The alpha parameter of the Hamiltonian.
            beta            : The beta parameter of the Hamiltonian.
        """
        self.parameter_string = repr(sorted(kwargs.iteritems()))
        
        self.symmetry_degree = kwargs['symmetry_degree']
        self.symmetry_class = kwargs['symmetry_class']
        assert fractions.gcd(self.symmetry_class, self.symmetry_degree), 'symmetry_class and symmetry_degree kwargs must be coprime.'
        assert 0 < self.symmetry_class < self.symmetry_degree, 'symmetry_class must be between 0 and symmetry_degree.'

        self.xy_mode_count = kwargs['xy_mode_count']
        assert self.xy_mode_count > 0, 'xy_mode_count must be positive.'
        # Make floor(self.xy_mode_count/2) positive modes and ceiling(self.xy_mode_count/2) negative modes.
        xy_mode_lower_bound = self.symmetry_class - self.symmetry_degree*((self.xy_mode_count+1)//2)
        xy_mode_upper_bound = self.symmetry_class + self.symmetry_degree*(self.xy_mode_count//2)
        self.xy_modes = range(xy_mode_lower_bound, xy_mode_upper_bound, self.symmetry_degree)
        assert len(self.xy_modes) == self.xy_mode_count

        self.sample_count = kwargs['sample_count']
        self.period = kwargs['period']
        self.sample_times = np.linspace(0.0, self.period, self.sample_count+1)[:-1] # Take all but the last element.

        assert len(self.sample_times) > 0, 'sample_times must be nonempty.'
        assert all(self.sample_times[i+1] > self.sample_times[i] for i in range(len(self.sample_times)-1)), 'sample_times must be a strictly increasing sequence.'
        assert self.period > self.sample_times[-1], 'period must be greater than last element of sample_times.'

        self.contraction = kwargs.get('contraction', 'np.einsum')
        assert self.contraction in ['np.einsum', 'tensor.contract', 'alt_einsum']

        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']

        start = time.time(); self.zeta_M,self.wz_modes,self.zeta_tensor = generate_zeta(self.xy_modes, self.contraction)#; print 'generate_zeta: {0} s'.format(time.time() - start)
        print 'len(sample_times) = {0}, len(xy_modes) = {1}, len(wz_modes) = {2}'.format(len(self.sample_times), len(self.xy_modes), len(self.wz_modes))
        start = time.time(); self.F_xy = realified_fourier.Transform(self.xy_modes, self.sample_times, self.period)#; print 'F_xy: {0} s'.format(time.time() - start)
        start = time.time(); self.F_wz = realified_fourier.Transform(self.wz_modes, self.sample_times, self.period)#; print 'F_wz: {0} s'.format(time.time() - start)
        start = time.time(); self.imag_projection_over_sample_times,self.imag_projection_matrix = generate_imag_projection_over_sample_times(self.sample_times)#; print 'generate_imag_projection_over_sample_times: {0} s'.format(time.time() - start)
        start = time.time(); self.imag_Q = generate_imag_Q(self.xy_modes, self.F_xy.omega)#; print 'generate_imag_Q: {0} s'.format(time.time() - start)
        start = time.time(); self.integrate_over_sample_times = generate_integrate_over_sample_times(self.sample_times, self.period, self.contraction)#; print 'generate_integrate_over_sample_times: {0} s'.format(time.time() - start)

        def xy_curve (R):
            if self.contraction == 'np.einsum':
                return np.einsum('ij,j', self.F_xy.samples_from_coeffs_matrix, R)
            elif self.contraction == 'tensor.contract':
                return tensor.contract('ij,j', self.F_xy.samples_from_coeffs_matrix, R, dtype=R.dtype)
            else:
                return alt_einsum('ij,j', self.F_xy.samples_from_coeffs_matrix, R)

        # start = time.time(); z_curve_tensor = np.einsum('ij,jk,klm', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix, self.zeta_tensor); print 'z_curve_tensor (with shape {0}): {1} s'.format(z_curve_tensor.shape, time.time() - start)
        start = time.time(); z_curve_tensor = np.einsum('ij,jkl', np.einsum('ij,jk', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix), self.zeta_tensor); print 'z_curve_tensor (with shape {0}): {1} s'.format(z_curve_tensor.shape, time.time() - start)
        def z_curve (R):
            if self.contraction == 'np.einsum':
                return np.einsum('ijk,j,k', z_curve_tensor, R, R)# + self.imag_Q(R)*self.sample_times
            elif self.contraction == 'tensor.contract':
                return tensor.contract('ijk,j,k', z_curve_tensor, R, R, dtype=R.dtype)# + self.imag_Q(R)*self.sample_times
            else:
                return alt_einsum('ijk,j,k', z_curve_tensor, R, R)# + self.imag_Q(R)*self.sample_times

        start = time.time(); xy_prime_curve_matrix = np.einsum('ij,jk', self.F_xy.samples_from_coeffs_matrix, self.F_xy.time_derivative_matrix); print 'xy_prime_curve_matrix (with shape {0}): {1} s'.format(xy_prime_curve_matrix.shape, time.time() - start)
        def xy_prime_curve (R):
            if self.contraction == 'np.einsum':
                return np.einsum('ij,j', xy_prime_curve_matrix, R)
            elif self.contraction == 'tensor.contract':
                return tensor.contract('ij,j', xy_prime_curve_matrix, R, dtype=R.dtype)
            else:
                return alt_einsum('ij,j', xy_prime_curve_matrix, R)

        # start = time.time(); z_prime_curve_tensor = np.einsum('ij,jk,kl,lmn', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix, self.F_wz.time_derivative_matrix, self.zeta_tensor); print 'z_prime_curve_tensor (with shape {0}): {1} s'.format(z_prime_curve_tensor.shape, time.time() - start)
        start = time.time(); z_prime_curve_tensor = np.einsum('ij,jkl', np.einsum('ij,jk', np.einsum('ij,jk', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix), self.F_wz.time_derivative_matrix), self.zeta_tensor); print 'z_prime_curve_tensor (with shape {0}): {1} s'.format(z_prime_curve_tensor.shape, time.time() - start)
        vector_of_ones = np.array([1.0 for _ in self.sample_times])
        def z_prime_curve (R):
            if self.contraction == 'np.einsum':
                return np.einsum('ijk,j,k', z_prime_curve_tensor, R, R)# + self.imag_Q(R)*vector_of_ones
            elif self.contraction == 'tensor.contract':
                return tensor.contract('ijk,j,k', z_prime_curve_tensor, R, R, dtype=R.dtype)# + self.imag_Q(R)*vector_of_ones
            else:
                return alt_einsum('ijk,j,k', z_prime_curve_tensor, R, R)# + self.imag_Q(R)*vector_of_ones

        z_prime_curve_dummy_tensor = np.zeros(self.sample_count)
        def z_prime_curve_dummy (R):
            return z_prime_curve_dummy_tensor

        self.position = lambda R : chi((xy_curve(R), z_curve(R)))
        # self.velocity = lambda R : chi((xy_prime_curve(R), z_prime_curve(R)))
        self.velocity = lambda R : chi((xy_prime_curve(R), z_prime_curve_dummy(R)))
        self.position_and_velocity = lambda R : eta((self.position(R), self.velocity(R)))
        self.lagrangian = generate_lagrangian(self.alpha, self.beta)
        self.lagrangian_v = generate_lagrangian_v(self.lagrangian)
        self.hamiltonian = generate_hamiltonian(self.alpha, self.beta)
        self.hamiltonian_v = generate_hamiltonian_v(self.hamiltonian)
        self.action = lambda R : self.integrate_over_sample_times(self.lagrangian_v(self.position_and_velocity(R)))
        self.Lambda = lambda R_lagmult : self.action(R_lagmult[:-1]) + R_lagmult[-1]*self.imag_Q(R_lagmult[:-1])

    def generate_symbolic_functions (self):
        def generate_variables ():
            R_lagmult_vars = np.ndarray((2*self.xy_mode_count+1,), dtype=object)
            R_lagmult_vars[:-1] = symbolic.tensor('R', (2*self.xy_mode_count,))
            R_lagmult_vars[-1] = symbolic.variable('lambduh')
            return R_lagmult_vars
        
        def compute_diff_and_print_progress (f, var, i, out_of):
            # sys.stdout.write('computing {0}th derivative out of {1} ... '.format(i, out_of))
            retval = load_cache_or_compute('cache/D_Lambda_{0}.{1}.pickle'.format(i, self.parameter_string), f.diff, var)
            # sys.stdout.write('complete.\n')
            return retval

        self.R_lagmult_vars = load_cache_or_compute('cache/R_lagmult_vars.{0}.pickle'.format(self.parameter_string), generate_variables)
        
        self.symbolic_Lambda = load_cache_or_compute('cache/Lambda.{0}.pickle'.format(self.parameter_string), lambda : self.Lambda(self.R_lagmult_vars))
        self.symbolic_D_Lambda = load_cache_or_compute('cache/D_Lambda.{0}.pickle'.format(self.parameter_string), lambda : np.array([compute_diff_and_print_progress(self.symbolic_Lambda, var, i, len(self.R_lagmult_vars)) for i,var in enumerate(self.R_lagmult_vars)]))
        self.symbolic_squared_L2_norm_D_Lambda = load_cache_or_compute('cache/objective_function.{0}.pickle'.format(self.parameter_string), lambda : sum(self.symbolic_Lambda.diff(var)**2 for var in self.R_lagmult_vars) / len(self.R_lagmult_vars))
        self.symbolic_constraint_function = load_cache_or_compute('cache/constraint_function.{0}.pickle'.format(self.parameter_string), lambda : self.imag_Q(self.R_lagmult_vars[:-1])**2)
    
    def generate_autowrapped_functions (self):
        import sympy.utilities.autowrap
        
        start = time.time()
        self.constraint_function = sympy.utilities.autowrap.autowrap(self.symbolic_constraint_function, args=self.R_lagmult_vars[:-1], backend='cython')
        print 'generating constraint_function (via autowrap) took {0} s.'.format(time.time() - start)
        
        start = time.time()
        self.objective_function = sympy.utilities.autowrap.autowrap(self.symbolic_squared_L2_norm_D_Lambda, args=self.R_lagmult_vars, backend='cython')
        print 'generating objective_function (via autowrap) took {0} s.'.format(time.time() - start)

def profile_ProblemContext ():
    symmetry_degree = 5
    symmetry_class = 2
    xy_mode_count_range = range(2,3+1)#range(2,10)
    sample_count_range = range(21,22+1)#range(16,48,8)
    period = 46.5
    alpha = 1.0
    beta = 16.0
    contractions = ['tensor.contract', 'alt_einsum']
    for contraction in contractions:
        for sample_count in sample_count_range:
            for xy_mode_count in xy_mode_count_range:
                try:
                    print '****************** contraction = {0}, sample_count = {1}, xy_mode_count = {2}'.format(contraction, sample_count, xy_mode_count)
                    pc = call_func_and_print_timing_info('ProblemContext', ProblemContext, symmetry_degree=symmetry_degree, symmetry_class=symmetry_class, xy_mode_count=xy_mode_count, sample_count=sample_count, period=period, alpha=alpha, beta=beta, contraction=contraction)
                    call_func_and_print_timing_info('pc.generate_symbolic_functions', pc.generate_symbolic_functions)
                    call_func_and_print_timing_info('pc.generate_autowrapped_functions', pc.generate_autowrapped_functions)
                except Exception as e:
                    print 'caught exception {0}'.format(repr(e))
                print ''
                print ''

# Initial conditions in form [alpha, beta, time, x,y,z, px, py, pz]:
#3-Fold:
#[1, 1/16, 273.5, 1, 0, 4 sqrt 3, 0, 1, 0]
def coreys_3_fold_curve ():
    import cmath
    import matplotlib.pyplot as plt
    import vector_field

    # Corey's 5-fold
    # initial_condition = [1.0, 0.0, 4.0*math.sqrt(3.0), 0.0, 1.0, 0.0]
    # period = 273.5
    # omega = cmath.exp(2.0j*math.pi/period)
    # print 'initial_condition = {0}'.format(initial_condition)

    alpha = 1.0
    beta = 1.0/16.0

    # H = generate_hamiltonian(alpha, beta)
    # print 'H(initial_condition) = {0}'.format(H(initial_condition))

    # hamiltonian_vector_field = generate_hamiltonian_vector_field(alpha, beta)

    # Xs,Ts = vector_field.compute_flow_curve(hamiltonian_vector_field, initial_condition, 0.0, period, sample_count)
    import heisenberg_dynamics
    Xs,Ts,period,sample_count = heisenberg_dynamics.compute_coreys_flow_curve()
    Xs = np.array(Xs)

    XY = np.ndarray((2*len(Xs),), dtype=float)
    XY[0::2] = Xs[:,0]
    XY[1::2] = Xs[:,1]

    X = Xs[:,0]
    Y = Xs[:,1]
    Z = Xs[:,2]

    import matplotlib.pyplot as plt

    plt.figure(1, figsize=(30,15))

    sp = plt.subplot(1,2,1)
    sp.set_title('(x,y) curve image')
    # plt.axes().set_aspect('equal')
    plt.plot(X,Y)
    
    sp = plt.subplot(1,2,2)
    sp.set_title('z(t)')
    plt.plot(Ts,Z)

    plt.savefig('3fold.png')



    return XY,period,sample_count,alpha,beta

def coreys_5_fold_curve (sample_count):
    import cmath
    import matplotlib.pyplot as plt
    import vector_field

    # Corey's 5-fold
    initial_condition = [1.0, 0.0, math.sqrt(3.0)/4.0, 0.0, 1.0, 0.0]
    period = 46.5
    omega = cmath.exp(2.0j*math.pi/period)
    print 'initial_condition = {0}'.format(initial_condition)

    alpha = 1.0
    beta = 16.0

    hamiltonian_vector_field = generate_hamiltonian_vector_field(alpha, beta)

    Xs,Ts = vector_field.compute_flow_curve(hamiltonian_vector_field, initial_condition, 0.0, period, sample_count)
    Xs = np.array(Xs)

    XY = np.ndarray((2*len(Xs),), dtype=float)
    XY[0::2] = Xs[:,0]
    XY[1::2] = Xs[:,1]

    # X = Xs[:,0]
    # Y = Xs[:,1]
    # Z = Xs[:,2]

    return XY,period,alpha,beta

    # # print Xs
    # X = [x for (x,_,_,_,_,_) in Xs]
    # Y = [y for (_,y,_,_,_,_) in Xs]
    # Z = [z for (_,_,z,_,_,_) in Xs]

    # plt.figure(1, figsize=(30,15))

    # sp = plt.subplot(1,2,1)
    # sp.set_title('(x,y) curve for RK4-solved dynamics')
    # # plt.axes().set_aspect('equal')
    # plt.plot(X,Y)
    
    # sp = plt.subplot(1,2,2)
    # sp.set_title('z(t) for RK4-solved dynamics')
    # plt.plot(Ts,Z)
    
    # # sp = plt.subplot(1,3,3)
    # # sp.set_title('H(t) for RK4-solved dynamics')
    # # plt.plot(Ts, [hamiltonian(X) for X in Xs])

    # plt.savefig('5fold.png')


#Initial conditions in form [alpha, beta, time, x,y,z, px, py, pz]:
#7-Fold:
#[2/pi, 16, 57.9, 1, 0, z, 0, 1, 0]  with z=sqrt{ pi^(-2) - 1/16 }
def coreys_7_fold_curve (sample_count):
    import cmath
    import matplotlib.pyplot as plt
    import vector_field

    # Corey's 5-fold
    c = math.sqrt(math.pi**-2 - 1.0/16.0)
    initial_condition = [1.0, 0.0, c, 0.0, 1.0, 0.0]
    period = 57.9
    omega = cmath.exp(2.0j*math.pi/period)
    print 'initial_condition = {0}'.format(initial_condition)

    alpha = 2.0/math.pi
    beta = 16.0

    hamiltonian_vector_field = generate_hamiltonian_vector_field(alpha, beta)

    Xs,Ts = vector_field.compute_flow_curve(hamiltonian_vector_field, initial_condition, 0.0, period, sample_count)
    Xs = np.array(Xs)

    XY = np.ndarray((2*len(Xs),), dtype=float)
    XY[0::2] = Xs[:,0]
    XY[1::2] = Xs[:,1]

    # X = Xs[:,0]
    # Y = Xs[:,1]
    # Z = Xs[:,2]

    # import matplotlib.pyplot as plt

    # plt.figure(1, figsize=(30,15))

    # sp = plt.subplot(1,2,1)
    # sp.set_title('(x,y) curve image')
    # # plt.axes().set_aspect('equal')
    # plt.plot(X,Y)
    
    # sp = plt.subplot(1,2,2)
    # sp.set_title('z(t)')
    # plt.plot(Ts,Z)

    # plt.savefig('7fold.png')



    return XY,period,alpha,beta

def main ():
    XY,period,sample_count,alpha,beta = coreys_3_fold_curve()

    # pc = ProblemContext(symmetry_degree=3, symmetry_class=2, xy_mode_count=60, sample_count=sample_count, period=period, alpha=alpha, beta=beta, contraction='np.einsum')
    # # pc = ProblemContext(symmetry_degree=5, symmetry_class=2, xy_mode_count=60, sample_count=sample_count, period=period, alpha=alpha, beta=beta, contraction='np.einsum')
    # # pc = ProblemContext(symmetry_degree=7, symmetry_class=2, xy_mode_count=60, sample_count=sample_count, period=period, alpha=alpha, beta=beta, contraction='np.einsum')
    # R = np.einsum('ij,j', pc.F_xy.coeffs_from_samples_matrix, XY)

    # # import scipy.optimize
    # # R = scipy.optimize.fmin(pc.imag_Q, R, maxfun=1000000)

    # P = pc.position(R)

    # import matplotlib.pyplot as plt

    # plt.figure(1, figsize=(30,15))

    # sp = plt.subplot(1,2,1)
    # sp.set_title('(x,y) curve image')
    # # plt.axes().set_aspect('equal')
    # plt.plot(P[:,0],P[:,1])
    
    # sp = plt.subplot(1,2,2)
    # sp.set_title('z(t)')
    # plt.plot(pc.sample_times,P[:,2])

    # plt.savefig('7fold.png')

    return 0

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    import scipy.optimize
    import sympy.utilities.autowrap

    # pc = ProblemContext(symmetry_degree=3, symmetry_class=1, xy_mode_count=7, sample_count=100, period=10.0, contraction='alt_einsum')
    # pc = ProblemContext(symmetry_degree=5, symmetry_class=2, xy_mode_count=10, sample_count=200, period=46.5, alpha=1.0, beta=16.0, contraction='alt_einsum')
    pc = call_func_and_print_timing_info('ProblemContext', ProblemContext, symmetry_degree=5, symmetry_class=2, xy_mode_count=15, sample_count=150, period=46.5, alpha=1.0, beta=16.0, contraction='tensor.contract')

    call_func_and_print_timing_info('pc.generate_symbolic_functions', pc.generate_symbolic_functions)
    call_func_and_print_timing_info('pc.generate_autowrapped_functions', pc.generate_autowrapped_functions)
    
    # Start with the 5-fold curve.
    XY = coreys_5_fold_curve(pc.sample_count)
    R = pc.F_xy.coeffs_from_samples_matrix.dot(XY)

    # R = np.random.randn(2*pc.xy_mode_count)
    # start = time.time()
    # R = scipy.optimize.fmin(lambda R : pc.constraint_function(*R), R, maxfun=100000000, disp=True)
    # print 'constraint optimization took {0} s.'.format(time.time() - start)

    start = time.time()
    R_lagmult = np.ndarray((2*pc.xy_mode_count+1,), dtype=float)
    R_lagmult[:-1] = R
    R_lagmult[-1] = 1.0 # Sort of arbitrary.
    R_lagmult = call_func_and_print_timing_info('optimize objective function', scipy.optimize.fmin, lambda R_lagmult : pc.objective_function(*R_lagmult), R_lagmult, disp=True, maxfun=100000000)#, callback=print_R)
    
    R = R_lagmult[:-1] # Extract all but the Lagrange multiplier.
    
    PV = call_func_and_print_timing_info('pc.position_and_velocity', pc.position_and_velocity, R)

    print 'action(R) = {0}'.format(pc.action(R))
    print 'imag_Q(R) = {0}'.format(pc.imag_Q(R))

    H = pc.hamiltonian_v(PV)
    L = pc.lagrangian_v(PV)

    import matplotlib.pyplot as plt

    plt.figure(1, figsize=(30,20))

    plt.subplot(2,3,1)
    plt.title('(x,y)')
    plt.plot(PV[:,0], PV[:,1])

    plt.subplot(2,3,2)
    plt.title('z(t), imag(Q(R)) = {0}'.format(pc.imag_Q(R)))
    plt.plot(pc.sample_times, PV[:,2])

    plt.subplot(2,3,3)
    plt.title('H(t)')
    plt.plot(pc.sample_times, H)

    plt.subplot(2,3,4)
    plt.title('(x\',y\')')
    plt.plot(PV[:,3], PV[:,4])

    plt.subplot(2,3,5)
    plt.title('z\'(t)')
    plt.plot(pc.sample_times, PV[:,5])

    plt.subplot(2,3,6)
    plt.title('L(t)')
    plt.plot(pc.sample_times, L)

    plt.savefig('dino.png')

if __name__ == '__main__':
    main()


