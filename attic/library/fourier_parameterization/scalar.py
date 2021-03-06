import itertools
import numpy as np
import tensor

class Scalar:
    def __init__ (self, frequencies, derivatives, closed_time_interval, dtype=float, tau=2.0*np.pi, cos=np.cos, sin=np.sin, double_nonzero_frequency_basis_functions=True):
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
        # This tensor is indexed as fourier_tensor[t,d,f,c]
        self.fourier_tensor = fourier_tensor = np.ndarray((T,D,F,2), dtype=dtype)
        for t,time in enumerate(half_open_time_interval):
            self.fourier_tensor[t,:,:,:] = Scalar._compute_partial_fourier_tensor(derivatives, frequencies, period, time, dtype, tau, cos, sin)

        # This is the linear transform projecting a time-sampled curve into the space of Fourier sums spanned
        # by the given frequencies.  This tensor is indexed as inverse_fourier_tensor[f,c,t].
        self.inverse_fourier_tensor = Scalar._compute_inverse_fourier_tensor(frequencies, period, half_open_time_interval, half_open_time_interval_deltas, dtype, tau, cos, sin, double_nonzero_frequency_basis_functions)

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
    def _compute_inverse_fourier_tensor (frequencies, period, half_open_time_interval, half_open_time_interval_deltas, dtype, tau, cos, sin, double_nonzero_frequency_basis_functions):
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
        if double_nonzero_frequency_basis_functions:
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
        import matplotlib.pyplot as plt
        import scipy.linalg
        import symbolic as sy
        import sympy as sp
        import sys
        import warnings

        warnings.filterwarnings('ignore', module='matplotlib')

        sys.stdout.write('fourier_parameterization.Scalar.test2()\n')

        def lerp (start, end, count):
            for i in xrange(count):
                yield (start*(count-1-i) + end*i)/(count-1)

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

            assert np.all(cos_coefficient_endomorphism == np.eye(*cos_coefficient_endomorphism.shape, dtype=int))
            assert np.all(cos_to_sin_coefficient_morphism == 0)
            assert np.all(sin_to_cos_coefficient_morphism == 0)
            expected_diagonal = np.array(map(lambda f:1 if f!=0 else 0, frequencies))
            assert np.all(sin_coefficient_endomorphism == np.diag(expected_diagonal))

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
            # period = sp.symbols('period')
            # tau = 2*sp.pi
            period = 10.0 # arbitrary
            tau = 2*np.pi
            # closed_time_interval = np.array(list(lerp(0, period, T+1)))
            closed_time_interval = np.linspace(0.0, period, T+1)

            fourier_parameterization = Scalar(frequencies, derivatives, closed_time_interval, dtype=float, tau=tau, cos=np.cos, sin=np.sin)

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

            # signal_endomorphism = tensor.contract('tdfc,fcT', fourier_parameterization.fourier_tensor, fourier_parameterization.inverse_fourier_tensor, output='tdT', dtype=object)
            signal_endomorphism = np.einsum('tdfc,fcT->tdT', fourier_parameterization.fourier_tensor, fourier_parameterization.inverse_fourier_tensor)
            assert signal_endomorphism.shape == (T,D,T)
            signal_endomorphism = signal_endomorphism[:,0,:]
            # signal_endomorphism = np.vectorize(sp.simplify)(signal_endomorphism)

            # The eigenvectors are the columns.
            eigenvalues,eigenvectors = scipy.linalg.eigh(signal_endomorphism)

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

            if actual_zero_eigenvalues == 1:
                zero_eigenvalue_filter = np.abs(eigenvalues) < 1.0e-12
                assert sum(zero_eigenvalue_filter) == 1
                zero_eigenvalue_index = list(zero_eigenvalue_filter).index(True)
                assert np.abs(eigenvalues[zero_eigenvalue_index]) < 1.0e-12
                return {
                    'T':T,
                    'period':period,
                    'highest_frequency':highest_frequency,
                    'signal_domain':fourier_parameterization.half_open_time_interval,
                    'eigenvector':eigenvectors[:,zero_eigenvalue_index],
                }
            else:
                return None

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
            zero_eigenvector_dv = []
            for T in xrange(4,16):
                d = test_signal_endomorphism(T)
                if d is not None:
                    zero_eigenvector_dv.append(d)

            row_count = len(zero_eigenvector_dv)
            col_count = 1
            fig,axes = plt.subplots(row_count, col_count, squeeze=False, figsize=(5*col_count, 3*row_count))

            for zero_eigenvector_index,zero_eigenvector_d in enumerate(zero_eigenvector_dv):
                axis = axes[zero_eigenvector_index][0]
                axis.set_title('zero eigenvector for T: {0}\nhighest frequency: {1}'.format(zero_eigenvector_d['T'], zero_eigenvector_d['highest_frequency']))
                signal_domain = zero_eigenvector_d['signal_domain']
                eigenvector = zero_eigenvector_d['eigenvector']
                axis.plot(signal_domain, eigenvector)
                axis.set_xlim(0.0, zero_eigenvector_d['period'])

            fig.tight_layout()
            filename = 'fourier_parameterization.scalar.test2.png'
            plt.savefig(filename, bbox_inches='tight')
            print 'wrote "{0}"'.format(filename)
            plt.close(fig)

    @staticmethod
    def test3 ():
        """
        Compute the Fourier coefficients of a given function and plot the resampled function.
        """
        import matplotlib.pyplot as plt
        import scipy.signal
        import sys

        def process_function (axis_row, closed_time_interval, frequencies, samples):
            assert len(samples.shape) == 1
            assert len(closed_time_interval) == samples.shape[0]+1

            derivatives = np.array([0])
            p = Scalar(frequencies, derivatives, closed_time_interval)
            fc = np.einsum('fct,t->fc', p.inverse_fourier_tensor, samples)
            assert fc.shape == (p.F,2)
            print 'mode coefficient for frequency 1:', fc[p.index_of_frequency(1)]
            reconstructed_samples = np.einsum('tdfc,fc->dt', p.fourier_tensor, fc)
            # print('reconstructed_samples.shape:', reconstructed_samples.shape)
            assert reconstructed_samples.shape == (1,p.T)
            reconstructed_samples = reconstructed_samples[0,:]

            max_reconstruction_error = np.max(np.abs(reconstructed_samples - samples))

            assert len(axis_row) >= 3

            axis = axis_row[0]
            axis.set_title('original function')
            axis.plot(p.half_open_time_interval, samples)

            axis = axis_row[1]
            axis.set_title('log abs of Fourier coefficients')
            axis.semilogy(p.frequencies, np.linalg.norm(fc, axis=1))

            axis = axis_row[2]
            axis.set_title('reconstructed function\nmax reconstruction error: {0}'.format(max_reconstruction_error))
            axis.plot(p.half_open_time_interval, reconstructed_samples)

        period = 10.0
        closed_time_interval = np.linspace(0.0, period, 1000)
        frequencies = np.linspace(0, 10, 11, dtype=np.int)

        row_count = 2
        col_count = 3
        fig,axis_row_v = plt.subplots(row_count, col_count, squeeze=False, figsize=(10*col_count,10*row_count))

        process_function(
            axis_row_v[0],
            closed_time_interval,
            frequencies,
            np.array([0.3+np.cos(2*np.pi/period*t) for t in closed_time_interval[:-1]])
        )

        process_function(
            axis_row_v[1],
            closed_time_interval,
            frequencies,
            scipy.signal.gaussian(len(closed_time_interval)-1, len(closed_time_interval)//10)
        )

        filename = 'fourier_parameterization.scalar.test3.png'
        plt.savefig(filename, bbox_inches='tight')
        print 'wrote "{0}"'.format(filename)
        plt.close(fig)

    @staticmethod
    def run_all_unit_tests ():
        Scalar.test1()
        Scalar.test2()
        Scalar.test3()

if __name__ == '__main__':
    Scalar.run_all_unit_tests()
