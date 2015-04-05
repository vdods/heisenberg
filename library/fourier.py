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
        self.dts_for_integral = linalg_util.FloatVector([sample_times[i+1] - sample_times[i] for i in range(len(sample_times)-1)])
        assert abs(self.dts_for_integral.sum() - self.period) < 1.0e-10, "The sum of dts should be equal to the period, up to numerical error."
        self.fourier_sum_matrix = \
            linalg_util.ComplexMatrix([[cmath.exp(self.i_omega*mode*sample_time) \
                                        for mode in self.modes] \
                                       for sample_time in self.sample_times])
        self.period_times_fourier_transform_matrix = \
            linalg_util.ComplexMatrix([[cmath.exp(-self.i_omega*mode*sample_time)*dt \
                                        for (sample_time,dt) in itertools.izip(self.sample_times,self.dts_for_integral)] \
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

    def product_of_signals (self, lhs_modes, lhs_coefficients, rhs_modes, rhs_coefficients):
        """
        If lhs_coefficients and rhs_coefficients represent functions A(t) and B(t), then this computes
        the coefficients of A(t)*B(t).  Note that this Transform object must have been produced from
        Transform.product_transform from the Transform objects corresponding to lhs_coefficients and
        rhs_coefficients.
        """
        import itertools
        return [sum(lhs_coefficient*rhs_coefficient \
                for ((lhs_mode,lhs_coefficient),(rhs_mode,rhs_coefficient)) \
                in itertools.product(itertools.izip(lhs_modes,lhs_coefficients), itertools.izip(rhs_modes,rhs_coefficients)) \
                if lhs_mode + rhs_mode == mode) for mode in self.modes]

    @staticmethod
    def product_transform (lhs, rhs):
        """
        If lhs and rhs are Transform objects representing signals A(t) and B(t),
        then this function computes the Fourier transform object which can exactly
        represent A(t)*B(t) -- in particular, this boils down to a discrete convolution
        of the modes of lhs and rhs.  The sample times for both lhs and rhs must be equal,
        but the modes don't have to be.
        """
        import itertools
        assert all(l == r for (l,r) in itertools.izip(lhs.sample_times,rhs.sample_times)), "The sample_times of lhs and rhs must be equal."

        import linalg_util
        import numpy
        # Form the set of modes -- the set of the sum of each pair of elements from the
        # respective sets of modes.
        modes = list(set([m0 + m1 for (m0,m1) in itertools.product(lhs.modes, rhs.modes)]))
        sample_times = numpy.append(lhs.sample_times, linalg_util.FloatVector([lhs.period]))
        return Transform(modes, sample_times)

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
        is the identity map.  Note that the map R*F : S -> S is an orthogonal linear
        projection, but is not necessarily the identity map.
        """

        def diff_norm_squared_for_composition (modes, sample_count):
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

        # Record start time for computing profiling information.
        import time
        start_time = time.time()

        # modes_upper_bounds = range(1,30+1)
        # sample_counts = range(3,100)
        modes_upper_bounds = range(1,3+1)
        sample_counts = range(3,10)

        epsilon = 1.0e-12
        epsilon_squared = epsilon**2
        test_case_count = 0
        failed_test_case_count = 0
        failed_test_cases = []
        for modes_upper_bound in modes_upper_bounds:
            for sample_count in sample_counts:
                if modes_upper_bound <= sample_count:
                    test_case_count += 1 # Increment the test counter
                    diff_norm_squared = diff_norm_squared_for_composition(range(modes_upper_bound),sample_count)
                    if diff_norm_squared >= epsilon_squared:
                        failed_test_case_count += 1
                        failed_test_cases.append({'modes_upper_bound':modes_upper_bound, 'sample_count':sample_count, 'diff_norm_squared':diff_norm_squared})
                    # assert diff_norm_squared < epsilon_squared, 'Composition F*R differs too much from the identity (norm squared of difference is {0}.'.format(diff_norm_squared)

        duration = time.time() - start_time
        timing_info = 'duration: {0}s, which was {1}s per test case.'.format(test_case_count, duration, duration/test_case_count)
        if failed_test_case_count == 0:
            print 'test_partial_inverse passed -- {0} test cases, {1}'.format(test_case_count, timing_info)
        else:
            print 'test_partial_inverse failed -- {0} failed out of {1} test cases, {2}'.format(failed_test_case_count, test_case_count, timing_info)
            print '    failed test cases:'
            for failed_test_case in failed_test_cases:
                print '    {0}'.format(failed_test_case)

    @staticmethod
    def test_product_of_signals ():
        """
        Tests product_transform and product_of_signals work correctly.  In particular, if A(t)
        and B(t) are signals with the same period and sampling, then product_of_signals called
        on their spectra (which could be different) produces the spectrum for A(t)*B(t), noting
        that the spectrum of the product is a discrete convolution of their modes.
        """

        import itertools
        import linalg_util
        import numpy
        period = 100.0
        sample_count = 100
        sample_times = numpy.linspace(0.0, period, num=sample_count+1)

        # TODO: generate a bunch of random sets of modes
        modes_choices = [range(0,mode_upper_bound+1) for mode_upper_bound in range(0,3+1)]
        for lhs_modes in modes_choices:
            lhs_transform = Transform(lhs_modes, sample_times)
            # Generate a random spectrum corresponding to lhs_modes
            lhs_coefficients = linalg_util.ComplexVector([numpy.random.random() + numpy.random.random()*1j for _ in lhs_modes])
            for rhs_modes in modes_choices:
                rhs_transform = Transform(rhs_modes, sample_times)
                # Generate a random spectrum corresponding to rhs_modes
                rhs_coefficients = linalg_util.ComplexVector([numpy.random.random() + numpy.random.random()*1j for _ in rhs_modes])

                product_transform = Transform.product_transform(lhs_transform, rhs_transform)

                # Verify that the sum of each of the pairs of modes is present in the product_transform's modes.
                for lhs_mode in lhs_modes:
                    for rhs_mode in rhs_modes:
                        assert lhs_mode+rhs_mode in product_transform.modes

                product_coefficients = product_transform.product_of_signals(lhs_modes, lhs_coefficients, rhs_modes, rhs_coefficients)
                # Verify that the product of signals is [almost] equal to the signal reconstructed
                # from the lhs and rhs spectra.
                lhs_signal = lhs_transform.sampled_sum_of(lhs_coefficients)
                rhs_signal = rhs_transform.sampled_sum_of(rhs_coefficients)
                actual_product_signal = linalg_util.ComplexVector([lhs_sample*rhs_sample for (lhs_sample,rhs_sample) in itertools.izip(lhs_signal,rhs_signal)])
                reconstructed_product_signal = product_transform.sampled_sum_of(product_coefficients)
                # print 'len(lhs_signal) = {0}, len(rhs_signal) = {1}, len(actual_product_signal) = {2}, len(reconstructed_product_signal = {3}'.format(len(lhs_signal), len(rhs_signal), len(actual_product_signal), len(reconstructed_product_signal))
                # print 'squared norm = {0}'.format(linalg_util.ComplexVectorNormSquared(actual_product_signal - reconstructed_product_signal))
                squared_error = linalg_util.ComplexVectorNormSquared(actual_product_signal - reconstructed_product_signal)
                assert squared_error < 1.0e-20

        # Test a particular case with known result -- opposing, period-1 exponentials.
        lhs_modes = [-1.0]
        rhs_modes = [1.0]
        lhs_transform = Transform(lhs_modes, sample_times)
        rhs_transform = Transform(rhs_modes, sample_times)
        lhs_coefficients = linalg_util.ComplexVector([1.0])
        rhs_coefficients = linalg_util.ComplexVector([1.0])
        product_transform = Transform.product_transform(lhs_transform, rhs_transform)
        assert product_transform.modes == [0.0]
        product_coefficients = product_transform.product_of_signals(lhs_modes, lhs_coefficients, rhs_modes, rhs_coefficients)
        assert product_coefficients == linalg_util.ComplexVector([1.0])

        print 'test_product_of_signals passed'
