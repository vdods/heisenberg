# NOTE: This changes a/b to produce a floating point approximation of that
# ratio, not the integer quotient.  For integer quotient, use a//b instead.
from __future__ import division

import fourier_parameterization
import numpy as np
import multiindex
import symbolic
import sympy
import tensor

class KeplerProblemContext:
    def __init__ (self, T=100, F=10, use_constraint=True):
        # Dimension of configuration space Q
        self.X = X = 2
        # Number of derivatives in phase space (2, i.e. position and velocity)
        self.D = D = 2
        # Indicates if the constraint should be used or not.
        self.use_constraint = use_constraint

        self.generate_functions()

        # self.fourier_curve_parameterization = FourierCurveParameterization(period=1.0, F=F, T=T, D=D)
        period = 1.0
        self.fourier_curve_parameterization = fourier_parameterization.Planar(np.array(range(-F,F+1)), np.array(range(D)), np.linspace(0.0, period, T+1))
        # self.riemann_sum_factor = self.fourier_curve_parameterization.period / self.fourier_curve_parameterization.T

        # 2 indicates that there are two coefficients for each (cos,sin) pair
        self.position_velocity_shape      = position_velocity_shape      = (X,D)
        self.fourier_coefficients_shape   = fourier_coefficients_shape   = self.fourier_curve_parameterization.fourier_coefficients_shape #(X,F,2)
        self.gravitational_constant_shape = gravitational_constant_shape = (1,)
        self.lagrange_multipliers_shape   = lagrange_multipliers_shape   = (1,)

        self.time_domain_parameter_count      = time_domain_parameter_count      = multiindex.prod(position_velocity_shape) + multiindex.prod(gravitational_constant_shape) + multiindex.prod(lagrange_multipliers_shape)
        self.frequency_domain_parameter_count = frequency_domain_parameter_count = multiindex.prod(fourier_coefficients_shape) + multiindex.prod(gravitational_constant_shape) + multiindex.prod(lagrange_multipliers_shape)

        # self.frequency_domain_parameters = frequency_domain_parameters = np.ndarray((frequency_domain_parameter_count,), dtype=float)
        # # Define names for the views.  Note that when assigning into the views, the
        # # [:] notation is necessary, otherwise el_form_fourier_coefficients_part and
        # # el_form_lagrange_multipliers_part will be reassigned to be different references
        # # altogether, and will no longer be views into euler_lagrange_form_buffer.
        # self.fc,self.g,self.lm = fc,g,lm = self.frequency_domain_views(self.frequency_domain_parameters)

        # fc.fill(0.0)
        # fc[self.fourier_curve_parameterization.index_of_frequency(0),0] = 0.1 # constant offset in x
        # fc[self.fourier_curve_parameterization.index_of_frequency(1),0] = 1.0 # cos for x
        # # fc[self.fourier_curve_parameterization.index_of_frequency(1),1] = 1.0 # sin for y

        # g[0] = 1.0

        # lm[0] = 0.0

        # # Defining this here avoids allocation in compute_euler_lagrange_form.
        # self.euler_lagrange_form_buffer = np.ndarray((frequency_domain_parameter_count,), dtype=float)
        # # Define names for the views.  Note that when assigning into the views, the
        # # [:] notation is necessary, otherwise el_form_fourier_coefficients_part and
        # # el_form_lagrange_multipliers_part will be reassigned to be different references
        # # altogether, and will no longer be views into euler_lagrange_form_buffer.
        # self.el_form_fc,self.el_form_g,self.el_form_lm = self.frequency_domain_views(self.euler_lagrange_form_buffer)

    def make_frequency_domain_parameters_and_views (self):
        frequency_domain_parameters = np.zeros((self.frequency_domain_parameter_count,), dtype=float)
        # Define names for the views.  Note that when assigning into the views, the
        # [:] notation is necessary, otherwise el_form_fourier_coefficients_part and
        # el_form_lagrange_multipliers_part will be reassigned to be different references
        # altogether, and will no longer be views into euler_lagrange_form_buffer.
        fd_fc,fd_g,fd_lm = self.frequency_domain_views(frequency_domain_parameters)
        return frequency_domain_parameters,fd_fc,fd_g,fd_lm

    def generate_functions (self, X=2):
        """
        Use sympy to define various functions and automatically compute their derivatives, crunching
        them down to (probably efficient) Python lambda functions.

        X is the dimension of the configuration space.
        """

        assert X > 0
        self.X = X

        # q is position
        q = symbolic.tensor('q', (X,))
        # v is velocity
        v = symbolic.tensor('v', (X,))
        # g is gravitational constant
        g = symbolic.variable('g')
        # lm is the lagrange multiplier
        lm = symbolic.variable('lm')
        # qv is the q and v variables in one array
        self.qv = qv = np.array(list(q) + list(v))
        # P is all variables
        self.P = P = np.array(list(q) + list(v) + [g, lm])

        # U is potential energy
        # U = -g / sympy.sqrt(np.sum(np.square(q)))
        U = -1 / sympy.sqrt(np.sum(np.square(q)))
        # K is kinetic energy -- NOTE: There is unit mass, so the momentum is the same as the velocity.
        # This fact is used in defining the Hamiltonian vector field.
        K = np.sum(np.square(v)) / 2
        # H is total energy (Hamiltonian)
        H = K + U
        # L is the difference in energy (Lagrangian)
        L = K - U
        # H_0 is the constant value which defines the constraint (that the Hamiltonian must equal that at all times)
        self.H_0 = H_0 = -1.8
        # This is the constraint.  The extra division is used to act as a metric on the lagrange multiplier coordinate.
        C = (H - H_0)**2 / 2 / 100

        # Construct the Hamiltonian vector field of H.
        dH = symbolic.D(H, qv)
        omega = np.zeros((2*X,2*X), dtype=np.int)
        omega[0:X,X:2*X] =  np.eye(X,X)
        omega[X:2*X,0:X] = -np.eye(X,X)
        # Symplectic gradient of H, aka the Hamiltonian vector field.
        X_H = tensor.contract('ij,j', omega, dH, dtype=object)

        # DL = symbolic.D(L, P)
        # # DH = symbolic.D(H, P)
        # DC = symbolic.D(C, P)

        # This is the integrand of the action functional
        Lambda_integrand = L #+ lm*C
        # This is the integrand of the first variation of the action
        DLambda_integrand = symbolic.D(Lambda_integrand, P)
        # This is the integrand for the constraint functional
        C_integrand = C

        # Solving the constrained optimization problem by minimizing the norm squared of DLambda.
        # DDLambda_integrand = symbolic.D(DLambda_integrand, P)
        Obj_integrand = (np.sum(np.square(DLambda_integrand))/2)#.simplify()
        DObj_integrand = symbolic.D(Obj_integrand, P)
        # print 'Obj_integrand =', Obj_integrand
        # print ''
        # print 'DObj_integrand =', DObj_integrand
        # print ''

        replacement_d = {'dtype=object':'dtype=float'}

        # self.L = symbolic.lambdify(L, P, replacement_d=replacement_d)
        # self.DL = symbolic.lambdify(DL, P, replacement_d=replacement_d)
        # self.DDL = symbolic.lambdify(DDL, P, replacement_d=replacement_d)
        # self.H = symbolic.lambdify(H, P, replacement_d=replacement_d)
        # self.DH = symbolic.lambdify(DH, P, replacement_d=replacement_d)

        self.H = symbolic.lambdify(H, P, replacement_d=replacement_d)
        self.X_H = symbolic.lambdify(X_H, qv, replacement_d=replacement_d)
        self.Lambda_integrand = symbolic.lambdify(Lambda_integrand, P, replacement_d=replacement_d)
        self.DLambda_integrand = symbolic.lambdify(DLambda_integrand, P, replacement_d=replacement_d)
        self.Obj_integrand = symbolic.lambdify(Obj_integrand, P, replacement_d=replacement_d)
        self.DObj_integrand = symbolic.lambdify(DObj_integrand, P, replacement_d=replacement_d)
        self.C_integrand = symbolic.lambdify(C_integrand, P, replacement_d=replacement_d)

    def time_domain_views (self, time_domain_parameters):
        """
        Returns a tuple (td_qv,td_g,td_lm), where each of the elements are views into:

            td_qv[x,d] : The position and velocity tensor.  The x index indexes the configuration space (i.e. x,y,z axis)
                         while the d index indexes the order of derivative (i.e. 0 is position, 1 is velocity).
            td_g[:]    : The scalar gravitational constant.
            td_lm[:]   : The Lagrange multiplier.

        Note that slice notation must be used to assign to these views, otherwise a new, unrelated local variable will be declared.
        """
        td_qv_count = multiindex.prod(self.position_velocity_shape)
        td_g_count = multiindex.prod(self.gravitational_constant_shape)
        td_lm_count = multiindex.prod(self.lagrange_multipliers_shape)
        assert time_domain_parameters.shape == (self.time_domain_parameter_count,)
        td_qv = time_domain_parameters[:td_qv_count].reshape(self.position_velocity_shape, order='F')
        td_g  = time_domain_parameters[td_qv_count:td_qv_count+td_g_count].reshape(self.gravitational_constant_shape, order='F')
        td_lm = time_domain_parameters[td_qv_count+td_g_count:].reshape(self.lagrange_multipliers_shape, order='F')
        return td_qv,td_g,td_lm

    def frequency_domain_views (self, frequency_domain_parameters):
        """
        Returns a tuple (fd_fc,fd_g,fd_lm), where each of the elements are views into:

            fd_fc[x,f,c] : The Fourier coefficients of the curve.  The x index indexes the configuration space (i.e. x,y,z axis),
                           the f index denotes the frequency, while c indexes which of cos or sin the coefficient is for (0 for cos, 1 for sin).
            fd_g[:]      : The scalar gravitational constant.
            fd_lm[:]     : The Lagrange multiplier.

        Note that slice notation must be used to assign to these views, otherwise a new, unrelated local variable will be declared.
        """
        fd_fc_count = multiindex.prod(self.fourier_coefficients_shape)
        fd_g_count = multiindex.prod(self.gravitational_constant_shape)
        fd_lm_count = multiindex.prod(self.lagrange_multipliers_shape)
        fd_fc = frequency_domain_parameters[:fd_fc_count].reshape(self.fourier_coefficients_shape, order='F')
        fd_g  = frequency_domain_parameters[fd_fc_count:fd_fc_count+fd_g_count].reshape(self.gravitational_constant_shape, order='F')
        fd_lm = frequency_domain_parameters[fd_fc_count+fd_g_count:].reshape(self.lagrange_multipliers_shape, order='F')
        return fd_fc,fd_g,fd_lm

    def curve_at_t (self, t, fc):
        return np.einsum('dfxc,fc->xd', self.fourier_curve_parameterization.fourier_tensor[t,:,:,:], fc)

    def curve (self, fc):
        return np.einsum('tdfxc,fc->txd', self.fourier_curve_parameterization.fourier_tensor, fc)

    def time_domain_variation_pullback_at_t (self, time_domain_parameter_variation, t):
        """Uses the Fourier-transform-parameterization of the curve to pull back a qv-g-lm vector to be a fc-g-lm vector."""
        assert time_domain_parameter_variation.shape == (self.time_domain_parameter_count,)
        td_qv,td_g,td_lm = self.time_domain_views(time_domain_parameter_variation)

        retval = np.ndarray((self.frequency_domain_parameter_count,), dtype=float)
        fd_fc,fd_g,fd_lm = self.frequency_domain_views(retval)

        fd_fc[:] = np.einsum('xd,dfxc->fc', td_qv, self.fourier_curve_parameterization.fourier_tensor[t,:,:,:,:])
        fd_g[:]  = td_g
        fd_lm[:] = td_lm

        return retval

    def time_domain_parameters_at_t (self, t, frequency_domain_parameters):
        retval = np.ndarray((self.time_domain_parameter_count,), dtype=float)
        td_qv,td_g,td_lm = self.time_domain_views(retval)
        fd_fc,fd_g,fd_lm = self.frequency_domain_views(frequency_domain_parameters)
        td_qv[:] = self.curve_at_t(t, fd_fc)
        td_g[:]  = fd_g
        td_lm[:] = fd_lm
        return retval

    def Lambda (self, frequency_domain_parameters):
        # return self.riemann_sum_factor * sum(self.Lambda_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters)) for t in xrange(self.fourier_curve_parameterization.T))
        return np.dot(
            self.fourier_curve_parameterization.half_open_time_interval_deltas,
            np.array([
                self.Lambda_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters))
                for t in xrange(self.fourier_curve_parameterization.T)
            ])
        )

    def DLambda_at_time (self, t, frequency_domain_parameters):
        return self.time_domain_variation_pullback_at_t(self.DLambda_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters)), t)

    def DLambda (self, frequency_domain_parameters, batch_t_v=None):
        if batch_t_v is None:
            batch_t_v = xrange(self.fourier_curve_parameterization.T)
            batch_size = self.fourier_curve_parameterization.T
        else:
            batch_size = len(batch_t_v)

        # return (self.fourier_curve_parameterization.period / batch_size) * sum(self.DLambda_at_time(t, frequency_domain_parameters) for t in batch_t_v)
        # return self.riemann_sum_factor * sum(self.DLambda_at_time(t, frequency_domain_parameters) for t in batch_t_v)
        return np.dot(
            self.fourier_curve_parameterization.half_open_time_interval_deltas,
            np.array([
                self.DLambda_at_time(t, frequency_domain_parameters)
                for t in batch_t_v
            ])
        )

    def Obj (self, frequency_domain_parameters):
        # return self.riemann_sum_factor * sum(self.Obj_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters)) for t in xrange(self.fourier_curve_parameterization.T))
        return np.dot(
            self.fourier_curve_parameterization.half_open_time_interval_deltas,
            np.array([
                self.Obj_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters))
                for t in xrange(self.fourier_curve_parameterization.T)
            ])
        )

    def DObj_at_time (self, t, frequency_domain_parameters):
        return self.time_domain_variation_pullback_at_t(self.DObj_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters)), t)

    def DObj (self, frequency_domain_parameters, batch_t_v=None):
        if batch_t_v is None:
            batch_t_v = xrange(self.fourier_curve_parameterization.T)
            batch_size = self.fourier_curve_parameterization.T
        else:
            batch_size = len(batch_t_v)

        # return (self.fourier_curve_parameterization.period / batch_size) * sum(self.DObj_at_time(t, frequency_domain_parameters) for t in batch_t_v)
        # return self.riemann_sum_factor * sum(self.DObj_at_time(t, frequency_domain_parameters) for t in batch_t_v)
        return np.dot(
            self.fourier_curve_parameterization.half_open_time_interval_deltas,
            np.array([self.DObj_at_time(t, frequency_domain_parameters) for t in batch_t_v])
        )

    def C (self, frequency_domain_parameters):
        # return self.riemann_sum_factor * sum(self.C_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters)) for t in xrange(self.fourier_curve_parameterization.T))
        return np.dot(
            self.fourier_curve_parameterization.half_open_time_interval_deltas,
            np.array([
                self.C_integrand(self.time_domain_parameters_at_t(t, frequency_domain_parameters))
                for t in xrange(self.fourier_curve_parameterization.T)
            ])
        )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import polynomial
    import scipy.integrate
    import scipy.signal
    import warnings

    warnings.filterwarnings('ignore', module='matplotlib')

    kpc = KeplerProblemContext()

    time = np.linspace(0.0, 6.0, 20000)
    qv_0 = np.array([1.5, 0.0, 0.0, 0.5])
    qv = scipy.integrate.odeint(lambda qv,t:kpc.X_H(qv), qv_0, time)

    # TODO: Use distance squared, then fit a parabola near the min and solve
    # for a better approximation of the min using subsample accuracy.
    jet_dist_squared = np.sum(np.square(qv - qv_0), axis=1)
    local_min_index_v = filter(lambda i:jet_dist_squared[i-1] > jet_dist_squared[i] and jet_dist_squared[i] < jet_dist_squared[i+1], xrange(1,len(jet_dist_squared)-2))
    print('locally minimizing jet-distances-squared:')
    for local_min_index in local_min_index_v:
        print('    index: {0:3}, time: {1:17}, jet_dist_squared: {2:17}'.format(local_min_index, time[local_min_index], jet_dist_squared[local_min_index]))
    assert len(local_min_index_v) > 0
    assert local_min_index_v[0] > 0
    estimated_period_time_index = local_min_index_v[0]
    assert 5 < estimated_period_time_index < len(time)-5
    if True:
        # Subtracting the center of the time window is necessary for the polynomial fit to be numerically stable.
        time_offset = time[estimated_period_time_index]
        time_window = time[estimated_period_time_index-5:estimated_period_time_index+6] - time_offset
        jet_dist_squared_window = jet_dist_squared[estimated_period_time_index-5:estimated_period_time_index+6]
        samples = np.vstack((time_window, jet_dist_squared_window)).T
        # Fit a quadratic polynomial
        coefficients = polynomial.fit(samples, 2)
        quadratically_approximated = np.vectorize(polynomial.python_function_from_coefficients(coefficients))(time_window)
        max_abs_approximation_error = np.max(np.abs(quadratically_approximated - jet_dist_squared_window))
        print 'max_abs_approximation_error for jet_dist_squared near minimum:', max_abs_approximation_error
        assert coefficients[2] > 0.0
        # Solve for the critical point, adding back in the time offset that was subtracted.
        estimated_period = -0.5 * coefficients[1] / coefficients[2] + time_offset
        print 'estimated_period:', estimated_period

    # Save the old time for plotting.
    old_time = time

    # Re-run the ODE integrator using the estimated period.
    time = np.linspace(0.0, estimated_period, 10001)
    # qv_0 is unchanged.
    qv = scipy.integrate.odeint(lambda qv,t:kpc.X_H(qv), qv_0, time)

    # Re-sample time and qv to be sparser.
    time = time[::10]
    assert len(time) == 1001
    qv = qv[::10,:]
    assert qv.shape == (1001,4)
    qv = qv[:-1,:]
    assert qv.shape == (1000,4)

    # estimated_period = time[estimated_period_time_index]

    H_v = np.apply_along_axis(kpc.H, 1, np.hstack((qv, np.zeros((qv.shape[0],2), dtype=float))))

    row_count = 2
    col_count = 3
    fig,axes = plt.subplots(row_count, col_count, squeeze=False, figsize=(8*col_count,8*row_count))

    axis = axes[0][0]
    axis.set_title('position')
    axis.scatter(qv[:,0], qv[:,1], s=1)
    axis.set_aspect('equal')
    axis.scatter([0], [0], color='black')

    axis = axes[0][1]
    axis.set_title('velocity')
    axis.scatter(qv[:,2], qv[:,3], s=1)
    axis.set_aspect('equal')
    axis.scatter([0], [0], color='black')

    axis = axes[1][0]
    axis.set_title('jet distance from initial condition\nestimated period: {0}'.format(estimated_period))
    for local_min_index in local_min_index_v:
        axis.axvline(old_time[local_min_index])
    axis.plot(old_time, jet_dist_squared)

    axis = axes[1][1]
    axis.set_title('Hamiltonian w.r.t. time\nmin-to-max range: {0}'.format(np.max(H_v)-np.min(H_v)))
    axis.plot(time[:-1], H_v)

    frequencies = np.linspace(-40, 40, 40+40+1, dtype=int)
    derivatives = np.array([0])
    # fp = fourier_parameterization.Planar(frequencies, derivatives, time[:estimated_period_time_index+1])
    # fc = np.einsum('fctx,tx->fc', fp.inverse_fourier_tensor, qv[:estimated_period_time_index,0:2])
    fp = fourier_parameterization.Planar(frequencies, derivatives, time)
    fc = np.einsum('fctx,tx->fc', fp.inverse_fourier_tensor, qv[:,0:2])
    reconstructed_q = np.einsum('tdfxc,fc->dtx', fp.fourier_tensor, fc)
    assert reconstructed_q.shape[0] == 1
    reconstructed_q = reconstructed_q[0,:,:]

    max_expected_error = 1.0e-3
    max_reconstruction_errors = np.max(np.abs(reconstructed_q-qv[:,0:2]), axis=0)
    print 'max_reconstruction_errors:', max_reconstruction_errors
    if max_expected_error is not None:
        assert np.all(max_reconstruction_errors <= max_expected_error), 'at least one component of max_reconstruction_errors ({0}) exceeded max_expected_error ({1})'.format(max_reconstruction_errors, max_expected_error)

    axis = axes[0][2]
    axis.set_title('reconstructed position\nmax reconstruction error: {0}'.format(np.max(max_reconstruction_errors)))
    axis.plot(qv[:,0], qv[:,1], lw=5, color='green', alpha=0.2)
    axis.scatter(reconstructed_q[:,0], reconstructed_q[:,1], s=1)
    axis.set_aspect('equal')
    axis.scatter([0], [0], color='black')

    axis = axes[1][2]
    axis.set_title('log abs of Fourier coefficients')
    axis.semilogy(fp.frequencies, np.linalg.norm(fc, axis=1), 'o')
    axis.semilogy(fp.frequencies, np.linalg.norm(fc, axis=1), lw=5, alpha=0.2)

    fig.tight_layout()
    filename = 'kepler.test.png'
    plt.savefig(filename)
    print('wrote "{0}"'.format(filename))
    plt.close(fig)

