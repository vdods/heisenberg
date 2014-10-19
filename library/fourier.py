class Transform:
    """
    The discretized Fourier transform for a particular sampling of a periodic function
    and a particular set of Fourier modes.
    """

    def __init__ (self, modes, sample_times):
        """
        sample_times must be an increasing sequence of numbers which sample the interval
        [0,period].  The last element of sample_times defines the period, which will be
        removed before assigning to self.sample_times.
        """
        assert sample_times[0] == 0.0, "sample_times[0] must be exactly 0.0."
        assert all(sample_times[i] < sample_times[i+1] for i in range(len(sample_times)-1)), "sample_times must be an increasing sequence."
        import cmath
        import itertools
        import linalg_util
        import math
        self.modes = modes
        self.period = sample_times[-1]
        self.omega = 2.0*math.pi/self.period
        self.i_omega = 1.0j*self.omega
        self.sample_times = linalg_util.FloatVector(sample_times[0:-1])
        dts_for_average = linalg_util.FloatVector([sample_times[i+1] - sample_times[i] for i in range(len(sample_times)-1)])
        assert abs(dts_for_average.sum() - self.period) < 1.0e-10, "The sum of dts should be equal to the period, up to numerical error."
        self.fourier_sum_matrix = \
            linalg_util.ComplexMatrix([[cmath.exp(self.i_omega*mode*sample_time) \
                                        for mode in self.modes] \
                                       for sample_time in self.sample_times])
        self.period_times_fourier_transform_matrix = \
            linalg_util.ComplexMatrix([[cmath.exp(-self.i_omega*mode*sample_time)*dt \
                                        for (sample_time,dt) in itertools.izip(self.sample_times,dts_for_average)] \
                                       for mode in self.modes])

    def coefficients_of (self, samples):
        """samples must be a complex-based numpy.array and correspond 1-to-1 with self.sample_times."""
        return self.period_times_fourier_transform_matrix.dot(samples) / self.period

    def sampled_sum_of (self, coefficients):
        """This uses self.sample_times to evaluate the sums.  coefficients must be a complex-based numpy.array and correspond exactly with self.modes."""
        return self.fourier_sum_matrix.dot(coefficients)

    def evaluate_sum_at_arbitrary_time (self, coefficients, t):
        """coefficients must be a complex-based numpy.array and correspond exactly with self.modes."""
        import cmath
        import itertools
        return sum(coefficient*cmath.exp(self.i_omega*mode*t) for (coefficient,mode) in itertools.izip(coefficients,self.modes))

    def resampled (self, sample_times):
        """
        Returns a fourier.Transform with the same modes as this one, but with the given sample_times.
        See fourier.Transform.__init__ for more documentation.
        """
        return Transform(self.modes, sample_times)

    @staticmethod
    def test (L, sample_count):
        import cmath
        import linalg_util
        import math
        import numpy
        # L = 10
        # modes = range(-L,L+1)
        modes = range(L)
        sample_times = numpy.linspace(0.0, 2.0*math.pi, num=sample_count+1)
        Ft = Transform(modes, sample_times)

        # Let S denote sample space.  Let C denote coefficient space.
        # Let F denote the linear map taking sample space to [Fourier] coefficient space.
        # Let R denote the linear map taking coefficient space to [reconstructed] sample space.

        F = Ft.period_times_fourier_transform_matrix / Ft.period
        R = Ft.fourier_sum_matrix

        # This composition gives the endomorphism of coordinate space
        # C_to_C = F.dot(R)
        S_to_S = R.dot(F)

        # print "|C_to_C - I|^2 = {0}".format(linalg_util.ComplexMatrixNormSquared(C_to_C - numpy.eye(len(C_to_C), dtype=complex)))
        # print "|S_to_S - I|^2 = {0}".format(linalg_util.ComplexMatrixNormSquared(S_to_S - numpy.eye(len(S_to_S), dtype=complex)))

        # return linalg_util.ComplexMatrixNormSquared(C_to_C - numpy.eye(len(C_to_C), dtype=complex))
        return linalg_util.ComplexMatrixNormSquared(S_to_S - numpy.eye(len(S_to_S), dtype=complex))
