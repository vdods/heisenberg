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
        # Number of derivatives specified
        self.D = D = len(derivatives)
        # Derivatives requested
        self.derivatives = derivatives
        # Times that the Fourier sum will be sampled at.
        self.closed_time_interval = closed_time_interval
        # Half-open interval without the right endpoint (the right endpoint defines the period).
        # This serves as the discretization of time at which the Fourier sum will be sampled.
        # This is a discrete sampling of a fundamental domain of the periodicity.
        self.half_open_time_interval = half_open_time_interval = closed_time_interval[:-1]
        # Number of time-samples in the fundamental domain
        self.T = T = len(half_open_time_interval)

        # This is the linear transform taking Fourier coefficients and producing a time-sampled curve.
        # I.e. the linear map from frequency domain to time domain.
        # 2 indicates that there are two coefficients for each (cos,sin) pair
        self.full_fourier_tensor = full_fourier_tensor = np.ndarray((T,D,F,2), dtype=dtype)
        for t,time in enumerate(half_open_time_interval):
            self.full_fourier_tensor[t,:,:,:] = Scalar.compute_fourier_tensor(derivatives, frequencies, period, time, dtype, tau, cos, sin)

    def sample (self, coefficient_tensor, dtype=float):
        assert coefficient_tensor.shape == (self.F,2)
        if dtype == object:
            return tensor.contract('tdfc,fc', self.full_fourier_tensor, coefficient_tensor, output='td', dtype=object)
        else:
            return np.einsum('tdfc,fc->td', self.full_fourier_tensor, coefficient_tensor)

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

        def lerp (start, end, count):
            for i in xrange(count):
                yield (start*(count-1-i) + end*i)/(count-1)

        frequencies = np.array([1,2,3,4,5])
        derivatives = np.array([0,1,2,3,4])
        period = sp.symbols('period')
        tau = 2*sp.pi
        closed_time_interval = np.array(list(lerp(0, period, 33)))

        s = Scalar(frequencies, derivatives, closed_time_interval, dtype=object, tau=tau, cos=sp.cos, sin=sp.sin)

        t = sp.symbols('t')
        coefficients = sy.tensor('c', (len(frequencies),2))
        basis_functions = np.array([[sp.cos(tau/period*f*t), sp.sin(tau/period*f*t)] for f in frequencies])
        f = np.dot(coefficients.flat, basis_functions.flat)

        s_sampled = s.sample(coefficients, dtype=object)

        expand_vec = np.vectorize(sp.expand)
        for d,derivative in enumerate(derivatives):
            sys.stdout.write('testing {0}th derivative...'.format(d))
            dth_derivative_of_f_sampled = np.array([f.diff(t,d).subs({t:sampled_t}) for sampled_t in s.half_open_time_interval])
            assert all(expand_vec(dth_derivative_of_f_sampled - s_sampled[:,d]) == 0)
            sys.stdout.write('passed.\n')

if __name__ == '__main__':
    Scalar.test()



# class Planar:
#     def __init__ (self, freq_v, period=1.0, T=50, D=1):
#         # Period of orbit
#         self.period = period
#         # The number of frequencies specified by freq_v.
#         self.freq_count = freq_count = len(freq_v)
#         # The frequencies appearing in this parameterization (e.g.. cos(tau/period*freq*t))
#         self.freq_v = freq_v
#         # Number of points in discretization Theta of fundamental domain of time parameter
#         self.T = T
#         # Number of derivatives in parameterization of curve
#         self.D = D

#         # A discrete, uniform sampling of the time range [0, period).
#         self.time_discretization = time_discretization = np.array([period*t/T for t in xrange(T)])
#         # This is the linear transform taking Fourier coefficients and producing a time-sampled curve.
#         # I.e. the linear map from frequency domain to time domain.
#         # 2 indicates that there are two coefficients for each (cos,sin) pair
#         self.full_fourier_tensor = full_fourier_tensor = np.ndarray((T,D,freq_count,2,2), dtype=float)
#         for t,time in enumerate(time_discretization):
#             self.full_fourier_tensor[t,:,:,:,:] = FourierCurveParameterization.compute_fourier_tensor(D, F, period, time)

#     @staticmethod
#     def compute_fourier_tensor (D, F, period, time):
#         complex_multiplication_tensor = np.zeros((2,2,2), dtype=float)
#         complex_multiplication_tensor[0,0,0] =  1.0
#         complex_multiplication_tensor[0,1,1] = -1.0
#         complex_multiplication_tensor[1,0,1] =  1.0
#         complex_multiplication_tensor[1,1,0] =  1.0

#         # 2 indicates that there are two coefficients; one for each element of a (cos,sin) pair
#         fourier_tensor = np.ndarray((D,F,2), dtype=float)
#         omega = tau/period
#         omega_f = omega*np.arange(F)
#         assert len(omega_f) == F
#         cos_sin_tensor = np.ndarray((F,2), dtype=float)
#         for d in xrange(D):
#             omega_f__d = omega_f**d
#             cos_sin_tensor[:,0] = (-1.0)**((d+1)//2) * omega_f__d * np.cos(omega_f*time)
#             cos_sin_tensor[:,1] = (-1.0)**(d//2)     * omega_f__d * np.sin(omega_f*time)
#             fourier_tensor[d,:,:,:] = np.einsum('ijk,fj->fik', complex_multiplication_tensor, cos_sin_tensor)
#         return fourier_tensor

# if __name__ == '__main__':
#     fpcp = FourierPlaneCurveParameterization(F=3)
