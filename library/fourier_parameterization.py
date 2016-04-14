import complex_multiplication
import itertools
import numpy as np
import tensor

# This could be faster, but for now, who cares.
def discrete_integral (array):
    retval = np.ndarray((len(array)+1,), dtype=array.dtype)
    retval[0] = 0.0
    retval[1:] = np.cumsum(array)
    return retval

class Scalar:
    def __init__ (self, frequencies, derivatives, closed_time_interval, dtype=float, tau=2.0*np.pi, cos=np.cos, sin=np.sin):
        # NOTE: This is designed to be able to work with any type, including e.g. sympy symbols.
        assert len(frequencies.shape) == 1
        assert len(derivatives.shape) == 1
        assert len(closed_time_interval.shape) == 1
        assert frequencies.shape[0] > 0
        assert derivatives.shape[0] > 0
        assert closed_time_interval.shape[0] >= 2
        assert len(frozenset(frequencies)) == len(frequencies), 'frequencies must contain unique values'
        assert len(frozenset(derivatives)) == len(derivatives), 'derivatives must contain unique values'

        # Period of orbit
        self.period = period = closed_time_interval[-1] - closed_time_interval[0]
        # Number of frequencies specified
        self.F = F = len(frequencies)
        # The frequencies of the basis (cos,sin) functions.
        self.frequencies = frequencies
        # The map giving an index for a valid frequency
        self.frequency_index_d = {frequency:i for i,frequency in enumerate(frequencies)}
        # Number of derivatives specified
        self.D = D = len(derivatives)
        # Derivatives requested
        self.derivatives = derivatives
        # The map giving an index for a valid derivative
        self.derivative_index_d = {derivative:i for i,derivative in enumerate(derivatives)}
        # Times that the Fourier sum will be sampled at.
        self.closed_time_interval = closed_time_interval
        # Half-open interval without the right endpoint (the right endpoint defines the period).
        # This serves as the discretization of time at which the Fourier sum will be sampled.
        # This is a discrete sampling of a fundamental domain of the periodicity.
        self.half_open_time_interval = half_open_time_interval = closed_time_interval[:-1]
        # Number of time-samples in the fundamental domain
        self.T = T = len(half_open_time_interval)
        # Compute the deltas between time interval points, which can be used e.g. in integrating
        # over the half-open time interval.  Note that this is not invariant under a reversal
        # of the time interval; TODO: compute a symmetric version of this.
        self.half_open_time_interval_deltas = half_open_time_interval_deltas = np.diff(closed_time_interval)
        # The shape of the coefficients tensor it expects for Scalar.sample.
        self.fourier_coefficients_shape = (F,2)

        # This is the linear transform taking Fourier coefficients and producing a time-sampled curve.
        # I.e. the linear map from frequency domain to time domain.
        # 2 indicates that there are two coefficients for each (cos,sin) pair
        self.fourier_tensor = fourier_tensor = np.ndarray((T,D,F,2), dtype=dtype)
        for t,time in enumerate(half_open_time_interval):
            self.fourier_tensor[t,:,:,:] = Scalar._compute_partial_fourier_tensor(derivatives, frequencies, period, time, dtype, tau, cos, sin)

        # This is the linear transform projecting a time-sampled curve into the space of Fourier
        # sums spanned by the given frequencies.
        self.inverse_fourier_tensor = Scalar._compute_inverse_fourier_tensor(frequencies, period, half_open_time_interval, half_open_time_interval_deltas, dtype, tau, cos, sin)

    def sample (self, coefficient_tensor, at_t=None, dtype=float):
        assert coefficient_tensor.shape == self.fourier_coefficients_shape, 'expected {0} but got {1}'.format(coefficient_tensor.shape)
        if at_t == None:
            return tensor.contract('tdfc,fc', self.fourier_tensor, coefficient_tensor, output='td', dtype=dtype)
        else:
            return tensor.contract('dfc,fc', self.fourier_tensor[t,:,:,:], coefficient_tensor, output='td', dtype=dtype)

    def index_of_frequency (self, frequency):
        return self.frequency_index_d[frequency]

    def index_of_derivative (self, derivative):
        return self.derivative_index_d[derivative]

    @staticmethod
    def _compute_partial_fourier_tensor (derivatives, frequencies, period, time, dtype, tau, cos, sin):
        """
        Computes the linear transformation taking Fourier coefficients and returning the X-jet of the corresponding time-series signal.
        In this case, the X-jet can be any set of derivatives (e.g. the 0th, or the 0th and 1st, or the 1st and 3rd, or any combination).
        """
        D = len(derivatives)
        F = len(frequencies)
        # 2 indicates that there are two coefficients; one for each element of a (cos,sin) pair
        partial_fourier_tensor = np.ndarray((D,F,2), dtype=dtype)
        omega = tau/period
        # The expression `omega_freq = omega*frequencies` sometimes produced a floating point constant multiple
        # for some reason, but the following expression seems to work more reliably.
        omega_freq = np.array([omega*frequency for frequency in frequencies])
        # print 'omega:', omega, 'type(omega):', type(omega), 'frequencies:', frequencies, 'frequencies.dtype:', frequencies.dtype, 'omega_freq:', omega_freq
        cos_sin_omega_freq = np.array([[cos(of*time), sin(of*time)] for of in omega_freq])
        for d,derivative in enumerate(derivatives):
            omega_freq__deriv = omega_freq**derivative
            partial_fourier_tensor[d,:,0] = (-1)**((derivative+1)//2) * omega_freq__deriv * cos_sin_omega_freq[:,derivative%2]
            partial_fourier_tensor[d,:,1] = (-1)**(derivative//2)     * omega_freq__deriv * cos_sin_omega_freq[:,(derivative+1)%2]
        # print 'partial_fourier_tensor[0,0,:]:'
        # print partial_fourier_tensor[0,0,:]
        return partial_fourier_tensor

    @staticmethod
    def _compute_inverse_fourier_tensor (frequencies, period, half_open_time_interval, half_open_time_interval_deltas, dtype, tau, cos, sin):
        """
        Computes the linear transformation taking a time-series signal and returning its Fourier coefficients for the specified frequencies.
        """
        assert len(half_open_time_interval) == len(half_open_time_interval_deltas)
        T = len(half_open_time_interval)
        F = len(frequencies)
        # print 'period:', period, 'half_open_time_interval:', half_open_time_interval, 'half_open_time_interval_deltas:', half_open_time_interval_deltas
        inverse_fourier_tensor = np.ndarray((F,2,T), dtype=dtype)
        omega = tau/period
        for f,frequency in enumerate(frequencies):
            for t,(time,delta) in enumerate(itertools.izip(half_open_time_interval,half_open_time_interval_deltas)):
                inverse_fourier_tensor[f,0,t] = cos(omega*frequency*time)*delta/period
                inverse_fourier_tensor[f,1,t] = sin(omega*frequency*time)*delta/period
        # Multiply the nonzero frequencies' components by the normalizing factor 2.
        inverse_fourier_tensor[frequencies!=0,:,:] *= 2
        # print 'inverse_fourier_tensor:'
        # print inverse_fourier_tensor
        return inverse_fourier_tensor

    @staticmethod
    def test1 ():
        import symbolic as sy
        import sympy as sp
        import sys

        sys.stdout.write('fourier_parameterization.Scalar.test1()\n')

        def lerp (start, end, count):
            for i in xrange(count):
                yield (start*(count-1-i) + end*i)/(count-1)

        D = 5
        frequencies = np.array([0,1,2,3,4,5])
        derivatives = np.array(range(D))
        period = sp.symbols('period')
        tau = 2*sp.pi
        T = 32
        closed_time_interval = np.array(list(lerp(0, period, T+1)))

        fourier_parameterization = Scalar(frequencies, derivatives, closed_time_interval, dtype=object, tau=tau, cos=sp.cos, sin=sp.sin)

        t = sp.symbols('t')
        coefficients = sy.tensor('c', (len(frequencies),2))
        basis_functions = np.array([[sp.cos(tau/period*f*t), sp.sin(tau/period*f*t)] for f in frequencies])
        f = np.dot(coefficients.flat, basis_functions.flat)

        s_sampled = fourier_parameterization.sample(coefficients, dtype=object)
        assert s_sampled.shape == (T,D)

        expand_vec = np.vectorize(sp.expand)
        for d,derivative in enumerate(derivatives):
            sys.stdout.write('testing {0}th derivative...'.format(d))
            dth_derivative_of_f_sampled = np.array([f.diff(t,d).subs({t:sampled_t}) for sampled_t in fourier_parameterization.half_open_time_interval])
            assert all(expand_vec(dth_derivative_of_f_sampled - s_sampled[:,d]) == 0)
            sys.stdout.write('passed.\n')

    @staticmethod
    def test2 ():
        import scipy.linalg
        import symbolic as sy
        import sympy as sp
        import sys

        sys.stdout.write('fourier_parameterization.Scalar.test2()\n')

        def lerp (start, end, count):
            for i in xrange(count):
                yield (start*(count-1-i) + end*i)/(count-1)

        def is_scalar_matrix (m, scalar):
            assert len(m.shape) == 2
            for i,j in itertools.product(xrange(m.shape[0]),xrange(m.shape[1])):
                if i == j:
                    if m[i,j] != scalar:
                        return False
                else:
                    if m[i,j] != 0:
                        return False
            return True

        def is_diagonal_matrix (m, diagonal_component_v):
            assert len(m.shape) == 2
            assert len(diagonal_component_v) == np.min(m.shape)
            for i,j in itertools.product(xrange(m.shape[0]),xrange(m.shape[1])):
                if i == j:
                    if m[i,j] != diagonal_component_v[i]:
                        return False
                else:
                    if m[i,j] != 0:
                        return False
            return True

        def test_spectrum_endomorphism (frequencies):
            F = len(frequencies)
            assert F > 0

            sys.stdout.write('test_spectrum_endomorphism(frequencies={0}) ... '.format(frequencies))

            D = 1
            derivatives = np.array(range(D))
            period = sp.symbols('period')
            tau = 2*sp.pi
            # By the Nyquist theorem, the number of time samples must be greater than twice the highest frequency.
            # But also we need there to be at least as many dimensions in the signal vector space as there
            # are in the spectrum space (or the spectrum_endomorphism can't hope to be the identity).
            T = max(2*F-np.sum(frequencies==0), 2*np.max(frequencies)+2) # TODO: Figure out why `np.max(frequencies)*2+2` works and `np.max(frequencies)*2+1` doesn't.
            closed_time_interval = np.array(list(lerp(0, period, T+1)))

            fourier_parameterization = Scalar(frequencies, derivatives, closed_time_interval, dtype=object, tau=tau, cos=sp.cos, sin=sp.sin)

            spectrum_endomorphism = tensor.contract('tdfc,FCt', fourier_parameterization.fourier_tensor, fourier_parameterization.inverse_fourier_tensor, output='FCdfc', dtype=object)
            assert spectrum_endomorphism.shape == (F,2,D,F,2)
            spectrum_endomorphism = spectrum_endomorphism[:,:,0,:,:]
            spectrum_endomorphism = np.vectorize(sp.simplify)(spectrum_endomorphism)

            cos_coefficient_endomorphism = spectrum_endomorphism[:,0,:,0]
            cos_to_sin_coefficient_morphism = spectrum_endomorphism[:,0,:,1]
            sin_to_cos_coefficient_morphism = spectrum_endomorphism[:,1,:,0]
            sin_coefficient_endomorphism = spectrum_endomorphism[:,1,:,1]

            # print 'frequencies:'
            # print frequencies
            # print 'T:', T

            # print 'cos_coefficient_endomorphism:'
            # print cos_coefficient_endomorphism
            # print 'cos_to_sin_coefficient_morphism:'
            # print cos_to_sin_coefficient_morphism
            # print 'sin_to_cos_coefficient_morphism:'
            # print sin_to_cos_coefficient_morphism
            # print 'sin_coefficient_endomorphism:'
            # print sin_coefficient_endomorphism

            assert is_scalar_matrix(cos_coefficient_endomorphism, 1)
            assert np.all(cos_to_sin_coefficient_morphism == 0)
            assert np.all(sin_to_cos_coefficient_morphism == 0)
            assert is_diagonal_matrix(sin_coefficient_endomorphism, np.array(map(lambda f:1 if f!=0 else 0, frequencies)))

            sys.stdout.write('passed.\n')

        def test_signal_endomorphism (T):
            assert T >= 1

            sys.stdout.write('test_signal_endomorphism(T={0}) ... '.format(T))

            D = 1
            derivatives = np.array(range(D)) # Only use 0th derivative (i.e. position)
            # Frequencies must be less than T//2, otherwise they'll count double because of aliasing.  The
            # frequency T//2 in particular will just show up as 0, because it has nodes at every sample.
            highest_frequency = (T-1)//2
            frequencies = np.array(range(highest_frequency+1))
            period = sp.symbols('period')
            tau = 2*sp.pi
            closed_time_interval = np.array(list(lerp(0, period, T+1)))

            fourier_parameterization = Scalar(frequencies, derivatives, closed_time_interval, dtype=object, tau=tau, cos=sp.cos, sin=sp.sin)

            # print ''
            # # print 'fourier_parameterization.fourier_tensor:'
            # for t in xrange(T):
            #     print 't:', t, 'time:', closed_time_interval[t]
            #     print 'fourier_parameterization.fourier_tensor[{0},0,f,c]:'.format(closed_time_interval[t])
            #     print fourier_parameterization.fourier_tensor[t,0,:,:]
            #     print 'fourier_parameterization.inverse_fourier_tensor[f,c,{0}]:'.format(closed_time_interval[t])
            #     print fourier_parameterization.inverse_fourier_tensor[:,:,t]
            #     print ''
            # print ''

            signal_endomorphism = tensor.contract('tdfc,fcT', fourier_parameterization.fourier_tensor, fourier_parameterization.inverse_fourier_tensor, output='tdT', dtype=object)
            assert signal_endomorphism.shape == (T,D,T)
            signal_endomorphism = signal_endomorphism[:,0,:].astype(float)
            # signal_endomorphism = np.vectorize(sp.simplify)(signal_endomorphism)

            eigenvalues = scipy.linalg.eigh(signal_endomorphism, eigvals_only=True)

            # print 'signal_endomorphism:'
            # print signal_endomorphism
            # print 'eigenvalues of signal_endomorphism:'
            # print eigenvalues

            expected_zero_eigenvalues = 1 if T%2==0 else 0
            actual_zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1.0e-12)
            expected_one_eigenvalues = T - expected_zero_eigenvalues
            actual_one_eigenvalues = np.sum(np.abs(eigenvalues-1) < 1.0e-12)
            assert actual_zero_eigenvalues == expected_zero_eigenvalues, 'expected {0} eigenvalues with value [near] 0, but got {1}'.format(expected_zero_eigenvalues, actual_zero_eigenvalues)
            assert actual_one_eigenvalues == expected_one_eigenvalues, 'expected {0} eigenvalues with value [near] 1, but got {1}'.format(expected_one_eigenvalues, actual_one_eigenvalues)

            sys.stdout.write('passed.\n')

        if True:
            for F in xrange(1,6):
                test_spectrum_endomorphism(np.array(range(F+1)))
            test_spectrum_endomorphism(np.array([0,2]))
            test_spectrum_endomorphism(np.array([0,3]))
            test_spectrum_endomorphism(np.array([1]))
            test_spectrum_endomorphism(np.array([2]))
            test_spectrum_endomorphism(np.array([1,3,5]))
            test_spectrum_endomorphism(np.array([0,3,4]))

        if True:
            for T in xrange(4,16):
                test_signal_endomorphism(T)

class Planar:
    def __init__ (self, frequencies, derivatives, closed_time_interval, dtype=float, tau=2.0*np.pi, cos=np.cos, sin=np.sin):
        # NOTE: This is designed to be able to work with any type, including e.g. sympy symbols.
        assert len(frequencies.shape) == 1
        assert len(derivatives.shape) == 1
        assert len(closed_time_interval.shape) == 1
        assert frequencies.shape[0] > 0
        assert derivatives.shape[0] > 0
        assert closed_time_interval.shape[0] >= 2

        # Period of orbit
        self.period = period = closed_time_interval[-1] - closed_time_interval[0]
        # Number of frequencies specified
        self.F = F = len(frequencies)
        # The frequencies of the basis (cos,sin) functions.
        self.frequencies = frequencies
        # The map giving an index for a valid frequency
        self.frequency_index_d = {frequency:i for i,frequency in enumerate(frequencies)}
        # Number of derivatives specified
        self.D = D = len(derivatives)
        # Derivatives requested
        self.derivatives = derivatives
        # The map giving an index for a valid derivative
        self.derivative_index_d = {derivative:i for i,derivative in enumerate(derivatives)}
        # Times that the Fourier sum will be sampled at.
        self.closed_time_interval = closed_time_interval
        # Half-open interval without the right endpoint (the right endpoint defines the period).
        # This serves as the discretization of time at which the Fourier sum will be sampled.
        # This is a discrete sampling of a fundamental domain of the periodicity.
        self.half_open_time_interval = half_open_time_interval = closed_time_interval[:-1]
        # Number of time-samples in the fundamental domain
        self.T = T = len(half_open_time_interval)
        # Compute the deltas between time interval points, which can be used e.g. in integrating
        # over the half-open time interval.  Note that this is not invariant under a reversal
        # of the time interval; TODO: compute a symmetric version of this.
        self.half_open_time_interval_deltas = np.diff(closed_time_interval)
        # The shape of the coefficients tensor it expects for Planar.sample.
        self.fourier_coefficients_shape = (F,2)

        scalar = Scalar(frequencies, derivatives, closed_time_interval, dtype=dtype, tau=tau, cos=cos, sin=sin)
        complex_multiplication_tensor = complex_multiplication.generate_complex_multiplication_tensor(dtype=dtype)

        # self.fourier_tensor = fourier_tensor = np.ndarray((T,D,F,2,2), dtype=dtype)
        self.fourier_tensor = tensor.contract('tdfc,xcp', scalar.fourier_tensor, complex_multiplication_tensor, output='tdfxp', dtype=dtype)
        assert self.fourier_tensor.shape == (T,D,F,2,2)

        # # This is the linear transform taking Fourier coefficients and producing a time-sampled curve.
        # # I.e. the linear map from frequency domain to time domain.
        # # 2 indicates that there are two coefficients for each (cos,sin) pair
        # self.fourier_tensor = fourier_tensor = np.ndarray((T,D,F,2,2), dtype=dtype)
        # for t,time in enumerate(half_open_time_interval):
        #     self.fourier_tensor[t,:,:,:] = Scalar._compute_partial_fourier_tensor(derivatives, frequencies, period, time, dtype, tau, cos, sin)

    def sample (self, coefficient_tensor, at_t=None, dtype=float, include_endpoint=False):
        assert coefficient_tensor.shape == self.fourier_coefficients_shape, 'expected {0} but got {1}'.format(coefficient_tensor.shape)
        if at_t == None:
            if include_endpoint:
                retval = np.ndarray((self.T+1,self.D,2), dtype=dtype)
            else:
                retval = np.ndarray((self.T,self.D,2), dtype=dtype)
            retval[:self.T,:,:] = tensor.contract('tdfxc,fc', self.fourier_tensor, coefficient_tensor, output='tdx', dtype=dtype)
            if include_endpoint:
                retval[self.T,:,:] = tensor.contract('dfxc,fc', self.fourier_tensor[0,:,:,:,:], coefficient_tensor, output='dx', dtype=dtype)
            return retval
        else:
            return tensor.contract('dfxc,fc', self.fourier_tensor[at_t,:,:,:,:], coefficient_tensor, output='dx', dtype=dtype)

    def index_of_frequency (self, frequency):
        return self.frequency_index_d[frequency]

    def index_of_derivative (self, derivative):
        return self.derivative_index_d[derivative]

    def arclength_reparameterization (self, fc, derivative_mask=None):
        """
        Returns a fourier_parameterization.Planar object such that if the curve for the
        Fourier coefficients fc is sampled [at its closed_time_interval values], then
        the points are spaced evenly in jet-space (or if derivative_mask is specified
        as a mask along the derivative axis of the curve, then only those derivatives'
        values will be used in determining the even spacing).

        This can be used for example to create a sampling of a Fourier-parameterized curve
        that is uniform in space and therefore more useful for numerically solving dynamics
        problems.
        """
        # TODO: Allow definition of inner product on curve jet space so that each derivative
        # can be weighted differently.  E.g. weight velocity less than position.  This will
        # take the form of contracting sample_deltas with itself using an inner product tensor
        # which carries the weights for each jet.  The reshaping done above won't happen.
        # sample_deltas = np.diff(curve[:,p.index_of_derivative(0),:], axis=0)

        # TODO: make this dtype-agnostic, so it can be used with sympy for example.

        curve = self.sample(fc, include_endpoint=True)
        if derivative_mask != None:
            curve = curve[:,derivative_mask,:]
        # Estimate the tangent vector field to the curve's at its sample points.   Note that
        # this particular definition is not invariant under reversal of the parameterization).
        tangent_vectors = np.einsum('ijk,i->ijk', np.diff(curve[:,:,:], axis=0), 1 / self.half_open_time_interval_deltas)
        speed = np.sqrt(np.einsum('ijk,ijk->i', tangent_vectors, tangent_vectors))
        assert np.all(np.abs(speed - np.apply_along_axis(np.linalg.norm, 1, tangent_vectors.reshape(curve.shape[0]-1,-1))) < 1.0e-8)
        # Compute the arclength and inverse arclength functions (via samplings of those functions).
        arclength_over_closed_interval = discrete_integral(speed)
        inv_arclength_function = np.vectorize(lambda s:np.interp(s, arclength_over_closed_interval, self.closed_time_interval))
        inv_arclength_over_closed_interval = inv_arclength_function(np.linspace(arclength_over_closed_interval[0], arclength_over_closed_interval[-1], len(self.closed_time_interval)))
        # Create a Planar object with the computed parameterization.
        # TODO: allow specification of frequencies, derivatives, dtype, etc.
        return Planar(self.frequencies, self.derivatives, inv_arclength_over_closed_interval)

    def arclength_reparameterization_2 (self, fc, derivative_mask=None):
        closed_time_interval = np.linspace(self.closed_time_interval[0], self.closed_time_interval[-1], len(self.closed_time_interval))
        p = Planar(self.frequencies, self.derivatives, closed_time_interval)
        return p.arclength_reparameterization(fc, derivative_mask)

    @staticmethod
    def test1 ():
        import symbolic as sy
        import sympy as sp
        import sys

        sys.stdout.write('fourier_parameterization.Planar.test1()\n')

        def lerp (start, end, count):
            for i in xrange(count):
                yield (start*(count-1-i) + end*i)/(count-1)

        frequencies = np.array([-4,-1,0,2,3])
        D = 5
        derivatives = np.array(range(D))
        period = sp.symbols('period')
        tau = 2*sp.pi
        T = 32
        closed_time_interval = np.array(list(lerp(0, period, T+1)))

        fourier_parameterization = Planar(frequencies, derivatives, closed_time_interval, dtype=object, tau=tau, cos=sp.cos, sin=sp.sin)

        complex_multiplication_tensor = complex_multiplication.generate_complex_multiplication_tensor(dtype=object)

        t = sp.symbols('t')
        coefficients = sy.tensor('c', (len(frequencies),2))
        basis_functions = np.array([[sp.cos(tau/period*f*t), sp.sin(tau/period*f*t)] for f in frequencies])
        f = tensor.contract('ijk,fj,fk', complex_multiplication_tensor, basis_functions, coefficients, output='i', dtype=object)

        s_sampled = fourier_parameterization.sample(coefficients, dtype=object)

        expand_vec = np.vectorize(sp.expand)
        diff_vec = np.vectorize(sp.diff)
        for d,derivative in enumerate(derivatives):
            sys.stdout.write('testing {0}th derivative...'.format(d))
            dth_derivative_of_f_sampled = np.array([np.vectorize(lambda x:x.subs({t:sampled_t}))(diff_vec(f,t,d)) for sampled_t in fourier_parameterization.half_open_time_interval])
            assert all((expand_vec(dth_derivative_of_f_sampled - s_sampled[:,d]) == 0).flat)
            sys.stdout.write('passed.\n')

    @staticmethod
    def test2 ():
        import matplotlib.pyplot as plt
        import sys

        sys.stdout.write('fourier_parameterization.Planar.test2()\n')

        frequencies = np.array([0,1])
        derivatives = np.array([0])
        period = 1.0
        T = 100
        closed_time_interval = np.linspace(0.0, period, T+1)
        p = Planar(frequencies, derivatives, closed_time_interval)
        omega = 2.0*np.pi/period
        fc = np.ndarray(p.fourier_coefficients_shape, dtype=float)

        fc.fill(0.0)
        fc[p.index_of_frequency(0),:] = [1.5, 2.5]
        expected_curve = np.array([[1.5, 2.5] for t in p.half_open_time_interval])
        curve = p.sample(fc)
        assert np.max(np.abs(expected_curve-curve[:,0,:])) < 1.0e-10

        fc.fill(0.0)
        fc[p.index_of_frequency(1),0] = 1.0
        expected_curve = np.array([np.cos(omega*p.half_open_time_interval), np.sin(omega*p.half_open_time_interval)]).T
        curve = p.sample(fc)
        assert np.max(np.abs(expected_curve-curve[:,0,:])) < 1.0e-10

        fc.fill(0.0)
        fc[p.index_of_frequency(1),1] = 1.0
        expected_curve = np.array([-np.sin(omega*p.half_open_time_interval), np.cos(omega*p.half_open_time_interval)]).T
        curve = p.sample(fc)
        assert np.max(np.abs(expected_curve-curve[:,0,:])) < 1.0e-10

        # plt.figure(1)
        # plt.plot(p.half_open_time_interval, curve[:,0,0], color='blue')
        # plt.plot(p.half_open_time_interval, curve[:,0,1], color='green')
        # plt.show()

        sys.stdout.write('passed.\n')

    @staticmethod
    def test3 ():
        """
        Generate a random periodic curve with 3-fold rotational symmetry, and then show that it
        can be reparameterized such that its sampling is uniform in Euclidean speed in its 0th
        derivative, 1th derivative, or a weighted combination of both.
        """
        import matplotlib.pyplot as plt
        import sys

        np.random.seed(4201)

        # Make a random closed curve that has a k-fold rotational symmetry.
        n = 3
        k = 3
        frequencies = np.array(range(1-k*n,1+k*n+1,k))
        derivatives = np.array([0,1])
        sample_count = 1024
        closed_time_interval = np.linspace(0.0, 1.0, sample_count+1)
        p = Planar(frequencies, derivatives, closed_time_interval)
        fc = np.random.randn(*p.fourier_coefficients_shape)

        def plot_curve (axis_row, p, fc, plot_name):
            curve = p.sample(fc, include_endpoint=True)
            assert curve.shape == (p.T+1,p.D,2)

            for d,derivative in enumerate(p.derivatives):
                axis = axis_row[d]
                axis.set_title('{0}\n{1}th derivative'.format(plot_name, derivative))
                axis.plot(curve[:,d,0], curve[:,d,1])
                # axis.plot((curve[-1,d,0],curve[0,d,0]), (curve[-1,d,1],curve[0,d,1]))
                axis.scatter(curve[:-1,d,0], curve[:-1,d,1], color='black', s=2)

            return curve

        fig,axis_row_v = plt.subplots(4, 2, squeeze=False, figsize=(20,40))
        plot_curve(axis_row_v[0], p, fc, 'uniform-in-time parameterization')

        inv_arclength_0_jet_p = p.arclength_reparameterization(fc, p.derivatives==0)
        plot_curve(axis_row_v[1], inv_arclength_0_jet_p, fc, 'uniform-in-0-jet-arclength parameterization')

        inv_arclength_1_jet_p = p.arclength_reparameterization(fc, p.derivatives==1)
        plot_curve(axis_row_v[2], inv_arclength_1_jet_p, fc, 'uniform-in-1-jet-arclength parameterization')

        inv_arclength_0_1_jet_p = p.arclength_reparameterization(fc)
        plot_curve(axis_row_v[3], inv_arclength_0_1_jet_p, fc, 'uniform-in-[0,1]-jet-arclength parameterization')

        # plt.show()
        filename = 'fourier_parameterization.test3.png'
        plt.savefig(filename, bbox_inches='tight')
        print 'wrote "{0}"'.format(filename)
        plt.close(fig)

if __name__ == '__main__':
    import sys

    # # Scalar.test1()
    Scalar.test2()
    sys.exit(0)

    # Planar.test2()
    # # sys.exit(0)

    # Scalar.test()
    # Planar.test1()

    Planar.test3()

