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
    def test_partial_inverse ():
        """
        Let M be the set of modes for which the Fourier coefficients will be computed.
        Let T be the set of times at which the signal function will be sampled, and
        assumed that T is uniformly distributed over the half-open interval [0,P), where
        P is the period.

        Let C be the space of Fourier coefficients corresponding to the modes M.  Note
        that C is [isomorphic to] the free complex vector space over M.  Let S be the
        space of function sample values over the set T.  Note that S is [isomorphic to]
        the free [real or] complex vector space over T.

        Let F be the linear map taking a sampled function to its M-spectrum.  This is the
        discrete analog of the Fourier transform of the function.

        Let R be the linear map taking a set of Fourier coefficients to the T-sampled
        signal that it represents.

        This test verifies that if dim(C) >= dim(S), then the composition F*R : C -> C
        is the identity map.  Note that the map R*F : S -> S is an linear projection,
        but is not necessarily the identity map.
        """

        def norm_squared_for_composition (modes, sample_count):
            import math
            import numpy
            sample_times = numpy.linspace(0.0, 2.0*math.pi, num=sample_count+1)
            Ft = Transform(modes, sample_times)

            # Let S denote sample space.  Let C denote coefficient space.
            # Let F denote the linear map taking sample space to [Fourier] coefficient space.
            # Let R denote the linear map taking coefficient space to [reconstructed] sample space.

            F = Ft.period_times_fourier_transform_matrix / Ft.period
            R = Ft.fourier_sum_matrix

            # This composition gives the endomorphism of coordinate space
            C_to_C = F.dot(R)

            import linalg_util
            return linalg_util.ComplexMatrixNormSquared(C_to_C - numpy.eye(len(C_to_C), dtype=complex))

        import sys
        modes_upper_bounds = range(1,30+1)
        sample_counts = range(3,100)

        epsilon = 1.0e-12
        epsilon_squared = epsilon**2
        for modes_upper_bound in modes_upper_bounds:
            for sample_count in sample_counts:
                if modes_upper_bound <= sample_count:
                    norm_squared = norm_squared_for_composition(range(modes_upper_bound),sample_count)
                    assert norm_squared < epsilon_squared, 'Composition F*R differs too much from the identity (norm squared of difference is {0}.'.format(norm_squared)
        print 'test_partial_inverse passed.' # TODO: print timing info
