import complex_multiplication
import fourier_parameterization
import numpy as np
import tensor

# This could be faster, but for now, who cares.
def discrete_integral (array):
    retval = np.ndarray((len(array)+1,), dtype=array.dtype)
    retval[0] = 0.0
    retval[1:] = np.cumsum(array)
    return retval

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

        scalar = fourier_parameterization.Scalar(frequencies, derivatives, closed_time_interval, dtype=dtype, tau=tau, cos=cos, sin=sin, double_nonzero_frequency_basis_functions=False)
        complex_multiplication_tensor = complex_multiplication.generate_complex_multiplication_tensor(dtype=dtype)

        self.fourier_tensor = tensor.contract('tdfp,xpc', scalar.fourier_tensor, complex_multiplication_tensor, output='tdfxc', dtype=dtype)
        assert self.fourier_tensor.shape == (T,D,F,2,2)

        # This particular contraction encodes the complex conjugation in the Hilbert space inner product
        # integral as a transpose of the complex multiplication tensor (in the first and last indices).
        self.inverse_fourier_tensor = tensor.contract('fpt,xpc', scalar.inverse_fourier_tensor, complex_multiplication_tensor, output='fctx', dtype=dtype)
        assert self.inverse_fourier_tensor.shape == (F,2,T,2)

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
        if derivative_mask is not None:
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
        filename = 'fourier_parameterization.planar.test3.png'
        plt.savefig(filename, bbox_inches='tight')
        print 'wrote "{0}"'.format(filename)
        plt.close(fig)

    @staticmethod
    def test4 ():
        """
        Compute the Fourier coefficients of a given curve and plot the resampled curve.
        """
        import matplotlib.pyplot as plt
        import sys

        def process_curve (axis_row, closed_time_interval, frequencies, points, max_expected_error=None):
            assert len(points.shape) == 2
            assert len(closed_time_interval) == points.shape[0]+1
            assert points.shape[1] == 2

            derivatives = np.array([0])
            p = Planar(frequencies, derivatives, closed_time_interval)
            fc = np.einsum('fctx,tx->fc', p.inverse_fourier_tensor, points)
            assert fc.shape == (p.F,2)
            print 'mode coefficient for frequency 1:', fc[p.index_of_frequency(1)]
            reconstructed_points = np.einsum('tdfxc,fc->dtx', p.fourier_tensor, fc)
            # print('reconstructed_points.shape:', reconstructed_points.shape)
            assert reconstructed_points.shape == (1,p.T,2)
            reconstructed_points = reconstructed_points[0,:,:]

            max_reconstruction_errors = np.max(np.abs(reconstructed_points-points), axis=0)
            print 'max_reconstruction_errors:', max_reconstruction_errors
            if max_expected_error is not None:
                assert np.all(max_reconstruction_errors <= max_expected_error), 'at least one component of max_reconstruction_errors ({0}) exceeded max_expected_error ({1})'.format(max_reconstruction_errors, max_expected_error)

            assert len(axis_row) >= 3

            axis = axis_row[0]
            axis.set_title('original curve')
            axis.scatter(points[:,0], points[:,1], s=1)
            center_point = np.mean(points, axis=0)
            axis.scatter(*center_point)
            print 'center_point:', center_point

            axis = axis_row[1]
            axis.set_title('original curve components\nblue:x, green:y')
            axis.scatter(closed_time_interval[:-1], points[:,0], color='blue', s=1)
            axis.scatter(closed_time_interval[:-1], points[:,1], color='green', s=1)

            axis = axis_row[2]
            axis.set_title('log abs of Fourier coefficients')
            axis.semilogy(p.frequencies, np.linalg.norm(fc, axis=1), 'o')
            axis.semilogy(p.frequencies, np.linalg.norm(fc, axis=1), lw=5, alpha=0.2)

            axis = axis_row[3]
            axis.set_title('reconstructed curve\nmax reconstruction errors: {0}'.format(max_reconstruction_errors))
            axis.plot(points[:,0], points[:,1], color='blue', lw=5, alpha=0.2)
            axis.scatter(reconstructed_points[:,0], reconstructed_points[:,1], s=1, color='black')
            reconstructed_center_point = fc[p.index_of_frequency(0),:]
            axis.scatter(*reconstructed_center_point)
            print 'reconstructed_center_point:', reconstructed_center_point

            axis = axis_row[4]
            axis.set_title('reconstructed curve components\nblue:x, green:y')
            axis.plot(closed_time_interval[:-1], points[:,0], color='blue', lw=5, alpha=0.2)
            axis.plot(closed_time_interval[:-1], points[:,1], color='green', lw=5, alpha=0.2)
            axis.scatter(closed_time_interval[:-1], reconstructed_points[:,0], color='blue', s=1)
            axis.scatter(closed_time_interval[:-1], reconstructed_points[:,1], color='green', s=1)

        period = 10.0
        closed_time_interval = np.linspace(0.0, period, 300)

        row_count = 2
        col_count = 5
        fig,axis_row_v = plt.subplots(row_count, col_count, squeeze=False, figsize=(10*col_count,10*row_count))

        frequencies = np.linspace(-1, 1, 3, dtype=np.int)
        process_curve(
            axis_row_v[0],
            closed_time_interval,
            frequencies,
            np.array([[0.3+np.cos(2*np.pi/period*t), 0.8+np.sin(2*np.pi/period*t)] for t in closed_time_interval[:-1]]),
            max_expected_error=1.0e-12
        )

        # Make a nearly-closed loop using Gaussians, and then wrap it back on itself to make it
        # closed and periodic (this is the outer sum part).
        fancy_curve = sum(
            np.exp(np.array([
                [-0.5*((t-0.4*period-offset)/(0.2*period))**2, -0.5*((t-0.6*period-offset)/(0.2*period))**2]
                for t in closed_time_interval[:-1]
            ]))
            for offset in np.linspace(-4*period, 4*period, 9)
        )

        frequencies = np.array([-10,-7,-4,-2,-1,0,1,2,4,8]) #np.linspace(-20, 20, 9, dtype=np.int)
        process_curve(
            axis_row_v[1],
            closed_time_interval,
            frequencies,
            fancy_curve,
            max_expected_error=0.0009
        )

        filename = 'fourier_parameterization.planar.test4.png'
        plt.savefig(filename, bbox_inches='tight')
        print 'wrote "{0}"'.format(filename)
        plt.close(fig)


    @staticmethod
    def run_all_unit_tests ():
        Planar.test1()
        Planar.test2()
        Planar.test3()
        Planar.test4()

if __name__ == '__main__':
    Planar.run_all_unit_tests()

