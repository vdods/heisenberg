import sys
sys.path.append('library')

import fractions
import math
import numpy as np
import realified_fourier
import symbolic
import tensor
import time

def generate_zeta (M, use_np_einsum):
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
        if use_np_einsum:
            return np.einsum('ijk,j,k', zeta_tensor, R, R)
        else:
            return tensor.contract('ijk,j,k', zeta_tensor, R, R, dtype=R.dtype)
    
    return zeta,L,zeta_tensor

def generate_imag_Q (M, period):
    # This is the imaginary part of 0.5*cmath.exp(2.0j*math.pi/period).
    half_imag_rotation = 0.5*math.sin(2.0*math.pi/period)

    def imag_Q (R):
        assert len(R.shape) == 1
        assert len(R) == 2*len(M), 'not enough input params.'
        return half_imag_rotation*sum(m*(R[2*i]**2 + R[2*i+1]**2) for i,m in enumerate(M))

    return imag_Q

def generate_integrate_over_sample_times (sample_times, period, use_np_einsum):
    assert len(sample_times) > 0, 'sample_times must be nonempty.'
    assert all(sample_times[i+1] > sample_times[i] for i in range(len(sample_times)-1)), 'sample_times must be a strictly increasing sequence.'
    assert period > sample_times[-1], 'period must be greater than last element of sample_times.'
    L = period - sample_times[0]
    integral_covector = np.array([sample_times[i+1] - sample_times[i] for i in range(len(sample_times)-1)] + [period - sample_times[-1]])
    assert len(integral_covector) == len(sample_times)
    def integrate_over_sample_times (X):
        if use_np_einsum:
            return np.einsum('j,j', integral_covector, X)
        else:
            return tensor.contract('j,j', integral_covector, X, dtype=X.dtype)
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

alpha = 2.0/math.pi
beta = 16.0

def hamiltonian (pv):
    assert len(pv) == 6
    mu = (pv[0]**2 + pv[1]**2)**2 + beta*pv[2]**2
    # First term is kinetic energy, second is potential.
    return 0.5*(pv[3]**2 + pv[4]**2) - alpha*mu**(-0.5)

def hamiltonian_v (PV):
    assert PV.shape[1] == 6
    return np.array([hamiltonian(pv) for pv in PV])

def lagrangian (pv):
    assert len(pv) == 6
    mu = (pv[0]**2 + pv[1]**2)**2 + beta*pv[2]**2
    # First term is kinetic energy, second is potential.
    return 0.5*(pv[3]**2 + pv[4]**2) + alpha*mu**(-0.5)

def lagrangian_v (PV):
    assert PV.shape[1] == 6
    return np.array([lagrangian(pv) for pv in PV])

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
            use_np_einsum   : A boolean indicating if np.einsum should be used instead of tensor.contract.  The default is True.
        """
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

        self.use_np_einsum = kwargs.get('use_np_einsum', True)

        start = time.time(); self.zeta_M,self.wz_modes,self.zeta_tensor = generate_zeta(self.xy_modes, self.use_np_einsum)#; print 'generate_zeta: {0} s'.format(time.time() - start)
        start = time.time(); self.F_xy = realified_fourier.Transform(self.xy_modes, self.sample_times, self.period)#; print 'F_xy: {0} s'.format(time.time() - start)
        start = time.time(); self.F_wz = realified_fourier.Transform(self.wz_modes, self.sample_times, self.period)#; print 'F_wz: {0} s'.format(time.time() - start)
        start = time.time(); self.imag_projection_over_sample_times,self.imag_projection_matrix = generate_imag_projection_over_sample_times(self.sample_times)#; print 'generate_imag_projection_over_sample_times: {0} s'.format(time.time() - start)
        start = time.time(); self.imag_Q = generate_imag_Q(self.xy_modes, self.F_xy.omega)#; print 'generate_imag_Q: {0} s'.format(time.time() - start)
        start = time.time(); self.integrate_over_sample_times = generate_integrate_over_sample_times(self.sample_times, self.period, self.use_np_einsum)#; print 'generate_integrate_over_sample_times: {0} s'.format(time.time() - start)

        def xy_curve (R):
            if self.use_np_einsum:
                return np.einsum('ij,j', self.F_xy.samples_from_coeffs_matrix, R)
            else:
                return tensor.contract('ij,j', self.F_xy.samples_from_coeffs_matrix, R, dtype=R.dtype)

        # start = time.time(); z_curve_tensor = np.einsum('ij,jk,klm', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix, self.zeta_tensor); print 'z_curve_tensor (with shape {0}): {1} s'.format(z_curve_tensor.shape, time.time() - start)
        start = time.time(); z_curve_tensor = np.einsum('ij,jkl', np.einsum('ij,jk', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix), self.zeta_tensor); print 'z_curve_tensor (with shape {0}): {1} s'.format(z_curve_tensor.shape, time.time() - start)
        def z_curve (R):
            if self.use_np_einsum:
                return np.einsum('ijk,j,k', z_curve_tensor, R, R) + self.imag_Q(R)*self.sample_times
            else:
                return tensor.contract('ijk,j,k', z_curve_tensor, R, R, dtype=R.dtype) + self.imag_Q(R)*self.sample_times

        start = time.time(); xy_prime_curve_matrix = np.einsum('ij,jk', self.F_xy.samples_from_coeffs_matrix, self.F_xy.time_derivative_matrix); print 'xy_prime_curve_matrix (with shape {0}): {1} s'.format(xy_prime_curve_matrix.shape, time.time() - start)
        def xy_prime_curve (R):
            if self.use_np_einsum:
                return np.einsum('ij,j', xy_prime_curve_matrix, R)
            else:
                return tensor.contract('ij,j', xy_prime_curve_matrix, R, dtype=R.dtype)

        # start = time.time(); z_prime_curve_tensor = np.einsum('ij,jk,kl,lmn', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix, self.F_wz.time_derivative_matrix, self.zeta_tensor); print 'z_prime_curve_tensor (with shape {0}): {1} s'.format(z_prime_curve_tensor.shape, time.time() - start)
        start = time.time(); z_prime_curve_tensor = np.einsum('ij,jkl', np.einsum('ij,jk', np.einsum('ij,jk', self.imag_projection_matrix, self.F_wz.samples_from_coeffs_matrix), self.F_wz.time_derivative_matrix), self.zeta_tensor); print 'z_prime_curve_tensor (with shape {0}): {1} s'.format(z_prime_curve_tensor.shape, time.time() - start)
        vector_of_ones = np.array([1.0 for _ in self.sample_times])
        def z_prime_curve (R):
            if self.use_np_einsum:
                return np.einsum('ijk,j,k', z_prime_curve_tensor, R, R) + self.imag_Q(R)*vector_of_ones
            else:
                return tensor.contract('ijk,j,k', z_prime_curve_tensor, R, R, dtype=R.dtype) + self.imag_Q(R)*vector_of_ones

        self.position = lambda R : chi((xy_curve(R), z_curve(R)))
        self.velocity = lambda R : chi((xy_prime_curve(R), z_prime_curve(R)))
        self.position_and_velocity = lambda R : eta((self.position(R), self.velocity(R)))
        self.action = lambda R : self.integrate_over_sample_times(lagrangian_v(self.position_and_velocity(R)))

def main ():
    import scipy.optimize
    import sympy.utilities.autowrap

    start = time.time()
    # pc = ProblemContext(symmetry_degree=3, symmetry_class=1, xy_mode_count=7, sample_count=100, period=10.0, use_np_einsum=False)
    pc = ProblemContext(symmetry_degree=3, symmetry_class=1, xy_mode_count=7, sample_count=100, period=10.0, use_np_einsum=False)
    print 'ProblemContext generated in {0} s.'.format(time.time() - start)

    # R = np.random.randn(2*pc.xy_mode_count)
    # print 'starting R = {0}'.format(R)
    # print 'imag_Q(R) = {0}'.format(pc.imag_Q(R))
    # def print_R (R):
    #     print 'R = {0}'.format(R)
    # start = time.time()
    # R = scipy.optimize.fmin(lambda R : pc.imag_Q(R)**2, R, disp=True)#, callback=print_R)
    # print 'optimization took {0} s.'.format(time.time() - start)

    R_vars = symbolic.tensor('R', (2*pc.xy_mode_count,))

    start = time.time()
    action = pc.action(R_vars)
    print 'computing symbolic expression for action took {0} s.'.format(time.time() - start)

    start = time.time()
    D_action_squared_L2_norm = sum(action.diff(r)**2 for r in R_vars) / len(R_vars)
    print 'computing symbolic expression for D_action took {0} s.'.format(time.time() - start)

    start = time.time()
    autowrapped_action = sympy.utilities.autowrap.autowrap(action, args=R_vars, backend='cython')
    print 'autowrapped_action computed in {0} s.'.format(time.time() - start)

    start = time.time()
    autowrapped_D_action_squared_L2_norm = sympy.utilities.autowrap.autowrap(D_action_squared_L2_norm, args=R_vars, backend='cython')
    print 'autowrapped_D_action_squared_L2_norm computed in {0} s.'.format(time.time() - start)

    start = time.time()
    R = np.random.randn(2*pc.xy_mode_count)
    R = scipy.optimize.fmin(lambda R : autowrapped_action(*R), R, disp=True, maxfun=10000)#, callback=print_R)
    print 'optimization took {0} s.'.format(time.time() - start)

    print 'action = {0}, D_action_squared_L2_norm = {1}.'.format(autowrapped_action(*R), autowrapped_D_action_squared_L2_norm(*R))

    start = time.time()
    PV = pc.position_and_velocity(R)
    print 'generated position and velocity in {0} s.'.format(time.time() - start)

    H = hamiltonian_v(PV)
    L = lagrangian_v(PV)

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


