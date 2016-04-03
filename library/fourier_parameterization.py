import complex_multiplication
import numpy as np
import tensor

class Scalar:
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
        # The shape of the coefficients tensor it expects for Scalar.sample.
        self.fourier_coefficients_shape = (F,2)

        # This is the linear transform taking Fourier coefficients and producing a time-sampled curve.
        # I.e. the linear map from frequency domain to time domain.
        # 2 indicates that there are two coefficients for each (cos,sin) pair
        self.full_fourier_tensor = full_fourier_tensor = np.ndarray((T,D,F,2), dtype=dtype)
        for t,time in enumerate(half_open_time_interval):
            self.full_fourier_tensor[t,:,:,:] = Scalar.compute_fourier_tensor(derivatives, frequencies, period, time, dtype, tau, cos, sin)

    def sample (self, coefficient_tensor, at_t=None, dtype=float):
        assert coefficient_tensor.shape == self.fourier_coefficients_shape, 'expected {0} but got {1}'.format(coefficient_tensor.shape)
        if at_t == None:
            return tensor.contract('tdfc,fc', self.full_fourier_tensor, coefficient_tensor, output='td', dtype=dtype)
        else:
            return tensor.contract('dfc,fc', self.full_fourier_tensor[t,:,:,:], coefficient_tensor, output='td', dtype=dtype)

    def index_of_frequency (self, frequency):
        return self.frequency_index_d[frequency]

    def index_of_derivative (self, derivative):
        return self.derivative_index_d[derivative]

    @staticmethod
    def compute_fourier_tensor (derivatives, frequencies, period, time, dtype, tau, cos, sin):
        D = len(derivatives)
        F = len(frequencies)
        # 2 indicates that there are two coefficients; one for each element of a (cos,sin) pair
        fourier_tensor = np.ndarray((D,F,2), dtype=dtype)
        omega = tau/period
        omega_freq = omega*frequencies
        if type(omega_freq) != np.ndarray:
            omega_freq = np.array([omega_freq])
        cos_sin_omega_freq = np.array([[cos(of*time), sin(of*time)] for of in omega_freq])
        for d,derivative in enumerate(derivatives):
            omega_freq__deriv = omega_freq**derivative
            # trig_func_0 = cos if derivative%2 == 0 else sin
            # trig_func_1 = sin if derivative%2 == 0 else cos
            fourier_tensor[d,:,0] = (-1)**((derivative+1)//2) * omega_freq__deriv * cos_sin_omega_freq[:,derivative%2] # np.array([trig_func_0(of*time) for of in omega_freq])
            fourier_tensor[d,:,1] = (-1)**(derivative//2)     * omega_freq__deriv * cos_sin_omega_freq[:,(derivative+1)%2] # np.array([trig_func_1(of*time) for of in omega_freq])
        return fourier_tensor

    @staticmethod
    def test ():
        import itertools
        import symbolic as sy
        import sympy as sp
        import sys

        sys.stdout.write('fourier_parameterization.Scalar.test()\n')

        def lerp (start, end, count):
            for i in xrange(count):
                yield (start*(count-1-i) + end*i)/(count-1)

        D = 5
        frequencies = np.array([1,2,3,4,5])
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

        # self.full_fourier_tensor = full_fourier_tensor = np.ndarray((T,D,F,2,2), dtype=dtype)
        self.full_fourier_tensor = tensor.contract('tdfc,xcp', scalar.full_fourier_tensor, complex_multiplication_tensor, output='tdfxp', dtype=dtype)
        assert self.full_fourier_tensor.shape == (T,D,F,2,2)

        # # This is the linear transform taking Fourier coefficients and producing a time-sampled curve.
        # # I.e. the linear map from frequency domain to time domain.
        # # 2 indicates that there are two coefficients for each (cos,sin) pair
        # self.full_fourier_tensor = full_fourier_tensor = np.ndarray((T,D,F,2,2), dtype=dtype)
        # for t,time in enumerate(half_open_time_interval):
        #     self.full_fourier_tensor[t,:,:,:] = Scalar.compute_fourier_tensor(derivatives, frequencies, period, time, dtype, tau, cos, sin)

    def sample (self, coefficient_tensor, at_t=None, dtype=float, include_endpoint=False):
        assert coefficient_tensor.shape == self.fourier_coefficients_shape, 'expected {0} but got {1}'.format(coefficient_tensor.shape)
        if at_t == None:
            if include_endpoint:
                retval = np.ndarray((self.T+1,self.D,2), dtype=dtype)
            else:
                retval = np.ndarray((self.T,self.D,2), dtype=dtype)
            retval[:self.T,:,:] = tensor.contract('tdfxc,fc', self.full_fourier_tensor, coefficient_tensor, output='tdx', dtype=dtype)
            if include_endpoint:
                retval[self.T,:,:] = tensor.contract('dfxc,fc', self.full_fourier_tensor[0,:,:,:,:], coefficient_tensor, output='dx', dtype=dtype)
            return retval
        else:
            return tensor.contract('dfxc,fc', self.full_fourier_tensor[at_t,:,:,:,:], coefficient_tensor, output='dx', dtype=dtype)

    def index_of_frequency (self, frequency):
        return self.frequency_index_d[frequency]

    def index_of_derivative (self, derivative):
        return self.derivative_index_d[derivative]

    @staticmethod
    def test1 ():
        import itertools
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
        import itertools
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
        import matplotlib.pyplot as plt
        import sys

        # This could be faster, but for now, who cares.
        def discrete_integral (array):
            retval = np.ndarray((len(array)+1,), dtype=array.dtype)
            retval[0] = 0.0
            retval[1:] = np.cumsum(array)
            return retval

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

        # fc = np.zeros(p.fourier_coefficients_shape, dtype=float)
        # fc[p.index_of_frequency(1),0] = 1

        def plot_curve (axis_row, p, fc):
            curve = p.sample(fc, include_endpoint=True)
            assert curve.shape == (p.T+1,p.D,2)

            for derivative_index,derivative in enumerate(p.derivatives):
                axis = axis_row[derivative_index]
                axis.set_title('{0}th derivative'.format(derivative))
                axis.plot(curve[:,derivative_index,0], curve[:,derivative_index,1])
                # axis.plot((curve[-1,derivative_index,0],curve[0,derivative_index,0]), (curve[-1,derivative_index,1],curve[0,derivative_index,1]))
                axis.scatter(curve[:-1,derivative_index,0], curve[:-1,derivative_index,1], color='black', s=2)

            return curve

        fig,axis_row_v = plt.subplots(2, 2, squeeze=False, figsize=(20,20))
        curve = plot_curve(axis_row_v[0], p, fc)

        # Compute the density of the curve's sample points.  This will be defined as
        # the pseudoinverse of the vector of distances from one sample to the next
        # (note that this is not invariant under reversal of the parameterization).
        # For now, only do this with the 0th derivative samples.
        deltas = np.diff(curve[:,p.index_of_derivative(0),:], axis=0)
        distances = np.apply_along_axis(np.linalg.norm, 1, deltas)
        arclength_over_closed_interval = discrete_integral(distances)
        inv_arclength_function = np.vectorize(lambda s:np.interp(s, arclength_over_closed_interval, p.closed_time_interval))
        inv_arclength_over_closed_interval = inv_arclength_function(np.linspace(arclength_over_closed_interval[0], arclength_over_closed_interval[-1], len(p.closed_time_interval)))

        # print 'p.closed_time_interval = {0}'.format(p.closed_time_interval)
        # print 'inv_arclength_over_closed_interval = {0}'.format(inv_arclength_over_closed_interval)

        inv_arclength_p = Planar(frequencies, derivatives, inv_arclength_over_closed_interval)

        plot_curve(axis_row_v[1], inv_arclength_p, fc)

        # plt.show()
        filename = 'fourier_parameterization.test3.png'
        plt.savefig(filename)
        print 'wrote "{0}"'.format(filename)
        plt.close(fig)

if __name__ == '__main__':
    import sys

    # Planar.test2()
    # # sys.exit(0)

    # Scalar.test()
    # Planar.test1()

    Planar.test3()

