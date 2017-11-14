import bisect
import numpy as np
from . import orbit_plot
import os
import scipy.fftpack
import scipy.signal
import sys
import time
from . import util
import vorpy
import vorpy.pickle
import vorpy.symplectic_integration

class ShootingMethodObjective:
    def __init__ (self, *, dynamics_context, preimage_qp_0=None, qp_0, t_max, t_delta, disable_salvage=False):
        self.__dynamics_context                 = dynamics_context
        self.preimage_qp_0                      = preimage_qp_0
        self.qp_0                               = qp_0
        self.__t_v                              = None
        self.__qp_v                             = None
        self.t_max                              = t_max
        self.t_delta                            = t_delta
        self.__Q_v                              = None
        self.__Q_global_min_index               = None
        self.__t_min                            = None
        self.__objective                        = None
        self.__resampled_time_d                 = {}                # keys are sample_count for the resampled curve.
        self.__resampled_flow_curve_d           = {}                # keys are sample_count for the resampled curve.
        self.__fft_xy_resampled_flow_curve_d    = {}                # keys are sample_count for the resampled curve.
        self.__fft_z_resampled_flow_curve_d     = {}                # keys are sample_count for the resampled curve.
        self.__symmetry_order_estimate          = None
        self.__symmetry_class_estimate          = None
        self.__symmetry_class_signal_v          = None
        self.__disable_salvage                  = disable_salvage
        self.flow_curve_was_salvaged            = False

    def configuration_space_dimension (self):
        return self.__dynamics_context.configuration_space_dimension()

    def flow_curve (self):
        if self.__qp_v is None:
            start_time = time.time() # TODO: Replace with Ticker usage

            t_v = np.arange(0.0, self.t_max, self.t_delta)
            order = 2
            #omega = vorpy.symplectic_integration.nonseparable_hamiltonian.heuristic_estimate_for_omega(delta=self.t_delta, order=order, c=10.0)
            # Want 2*omega*t_delta = pi/2, meaning that omega = pi/(4*t_delta)
            omega = np.pi/(4*self.t_delta)
            assert np.allclose(2*omega*self.t_delta, np.pi/2)
            try:
                qp_v = vorpy.symplectic_integration.nonseparable_hamiltonian.integrate(
                    initial_coordinates=self.qp_0,
                    t_v=t_v,
                    dH_dq=self.__dynamics_context.dH_dq,
                    dH_dp=self.__dynamics_context.dH_dp,
                    order=order,
                    omega=omega
                )
                self.flow_curve_was_salvaged        = False
            except vorpy.symplectic_integration.exceptions.SalvagedResultException as e:
                print('salvaged results from exception encountered in nonseparable_hamiltonian.integrate: {0}'.format(e))
                original_step_count             = len(t_v)
                qp_v                            = e.salvaged_qp_v
                t_v                             = e.salvaged_t_v
                self.flow_curve_was_salvaged    = True

                # Set these here so that this ShootingMethodObjective is fully defined for use in OrbitPlot.plot_curve.
                self.__t_v  = t_v
                self.__qp_v = qp_v

                if not self.__disable_salvage:
                    # TEMP: Plot this salvaged curve in order to diagnose what went wrong
                    curve_description = 'salvaged curve - {0} steps out of {1}'.format(e.salvaged_qp_v.shape[0], original_step_count)
                    op = orbit_plot.OrbitPlot(curve_description_v=[curve_description], quantity_to_plot_v=orbit_plot.default_quantity_to_plot_v)
                    op.plot_curve(curve_description=curve_description, smo=self)
                    op.savefig_and_clear(
                        filename=os.path.join(
                            'heisenberg.custom_plot', # TODO: Specify salvaged result directory
                            'salvaged.obj:{0:.4e}.t_delta:{1:.3e}.t_max:{2:.3e}.ic:{3}.png'.format(
                                self.objective(),
                                self.t_delta,
                                self.t_max,
                                util.ndarray_as_single_line_string(self.qp_0)
                            )
                        )
                    )

            print('integration took {0} seconds'.format(time.time() - start_time))

            self.__t_v  = t_v
            self.__qp_v = qp_v
            assert self.__qp_v is not None
        return self.__qp_v

    def resampled_time (self, *, sample_count):
        if sample_count not in self.__resampled_time_d:
            t_min = self.t_min()
            t_v = self.t_v()

            rt  = np.linspace(t_v[0], t_min, sample_count, endpoint=False) # Resampled time

            self.__resampled_time_d[sample_count]       = rt

            assert sample_count in self.__resampled_time_d
        return self.__resampled_time_d[sample_count]

    def resampled_flow_curve (self, *, sample_count):
        """
        This uses linear interpolation -- could use the fact that points in flow_curve lie in a phase space
        and therefore the tangent vectors at each point are known (though the momenta) in order to interpolate
        the positions using cubic Bezier, and then interpolate the momenta accordingly.
        """
        if sample_count not in self.__resampled_flow_curve_d:
            flow_curve = self.flow_curve()
            t_min = self.t_min()
            t_v = self.t_v()

            def linearly_interpolated_flow_curve_at (t):
                assert t_v[0] <= t_v[-1], 't (which is {0}) is out of bounds (which are [{1}, {2}])'.format(t, t_v[0], t_v[-1])

                if t == t_v[0]:
                    return flow_curve[0]
                elif t == t_v[-1]:
                    return flow_curve[-1]

                i = bisect.bisect_left(t_v, t)
                assert 0 < i < len(t_v)
                parameter = (t - t_v[i-1]) / (t_v[i] - t_v[i-1])
                assert 0.0 <= parameter <= 1.0
                return (1.0-parameter)*flow_curve[i-1,...] + parameter*flow_curve[i,...]

            assert flow_curve.shape[1:] == (2,3)
            rt  = self.resampled_time(sample_count=sample_count)
            rfc = np.ndarray((sample_count,2,3), dtype=flow_curve.dtype)
            for i,t in enumerate(rt):
                rfc[i,...] = linearly_interpolated_flow_curve_at(t)

            self.__resampled_flow_curve_d[sample_count] = rfc

            assert sample_count in self.__resampled_flow_curve_d
        return self.__resampled_flow_curve_d[sample_count]

    def fft_xy_resampled_flow_curve (self, *, sample_count=1024):
        if sample_count not in self.__fft_xy_resampled_flow_curve_d:
            if sample_count != scipy.fftpack.next_fast_len(sample_count):
                print('WARNING in ShootingMethodObjective.fft_xy_resampled_flow_curve: Using SLOW sample_count (which is {0}); scipy.fftpack.next_fast_len(sample_count) is {1}.  A power of 2 is the ideal choice.'.format(sample_count, scipy.fftpack.next_fast_len(sample_count)))

            rfc = self.resampled_flow_curve(sample_count=sample_count)
            assert rfc.shape[0] == sample_count
            # Take only the xy component of the curve, represented as complex numbers x+i*y
            xy_rfc = rfc[:,0,0] + 1j*rfc[:,0,1]
            fft_xy_rfc = scipy.fftpack.fft(xy_rfc)

            self.__fft_xy_resampled_flow_curve_d[sample_count] = fft_xy_rfc

            assert sample_count in self.__fft_xy_resampled_flow_curve_d
        return self.__fft_xy_resampled_flow_curve_d[sample_count]

    def fft_z_resampled_flow_curve (self, *, sample_count=1024):
        if sample_count not in self.__fft_z_resampled_flow_curve_d:
            if sample_count != scipy.fftpack.next_fast_len(sample_count):
                print('WARNING in ShootingMethodObjective.fft_z_resampled_flow_curve: Using SLOW sample_count (which is {0}); scipy.fftpack.next_fast_len(sample_count) is {1}.  A power of 2 is the ideal choice.'.format(sample_count, scipy.fftpack.next_fast_len(sample_count)))

            rfc = self.resampled_flow_curve(sample_count=sample_count)
            assert rfc.shape[0] == sample_count
            # Take only the xy component of the curve, represented as complex numbers x+i*y
            z_rfc = rfc[:,0,2]
            fft_z_rfc = scipy.fftpack.fft(z_rfc)

            self.__fft_z_resampled_flow_curve_d[sample_count] = fft_z_rfc

            assert sample_count in self.__fft_z_resampled_flow_curve_d
        return self.__fft_z_resampled_flow_curve_d[sample_count]

    def symmetry_order_estimate (self):
        """
        Use Fourier analysis to estimate the order of the symmetry of the curve.  Note that the curve
        may not be periodic, but the order of "the nearby periodic orbit" may still be estimated.
        """

        # sample_count specifies the number of samples to use in the FFT, and for now should probably
        # be picked to be larger than 2*k for the largest k we're reasonably dealing with (say around
        # k=20).  Note that this is still experimental.
        sample_count = 1024

        if self.__symmetry_order_estimate is None:
            fft_z_rfc = self.fft_z_resampled_flow_curve(sample_count=sample_count)

            # TODO: Probably need to use a windowing function to deal with curves that don't close up

            # Note that because our signal is real-valued, the FFT is an even function.  Thus we
            # can look at only the nonnegative modes to determine the desired information.
            # nonnegative_mode_v should.  TODO: Probably don't need nonnegative_mode_v.
            nonnegative_mode_v = scipy.fftpack.fftfreq(sample_count, d=1.0/sample_count)[:(sample_count+1)//2]
            assert np.allclose(nonnegative_mode_v, np.round(nonnegative_mode_v)), 'nonnegative_mode_v should contain integral values'
            nonnegative_mode_v = np.round(nonnegative_mode_v).astype(int)
            nonnegative_mode_fft_v = fft_z_rfc[:(sample_count+1)//2]
            abs_nonnegative_mode_fft_v = np.abs(nonnegative_mode_fft_v)

            # Calculate the strongest mode -- put into descending order -- strongest first.
            strength_sorted_mode_index_v = np.argsort(abs_nonnegative_mode_fft_v)[::-1]
            assert len(strength_sorted_mode_index_v) == len(nonnegative_mode_v)
            print('strength-sorted modes (only the first few):')
            for strength_sorted_mode_index in strength_sorted_mode_index_v[:5]:
                print('    {0} : {1:e} (freq = {2})'.format(strength_sorted_mode_index, abs_nonnegative_mode_fft_v[strength_sorted_mode_index], nonnegative_mode_v[strength_sorted_mode_index]))

            # TODO: Figure out some threshold for salience of strongest mode to second strongest mode.
            # TODO: Maybe also compute an "estimate salience" value
            self.__symmetry_order_estimate = strength_sorted_mode_index_v[0]
            assert self.__symmetry_order_estimate is not None
        return self.__symmetry_order_estimate

    def symmetry_class_estimate (self):
        """
        Use Fourier analysis to estimate the class of the symmetry of the curve.  This takes the form
        class:order, where class < order and class is coprime to order.  These are therefore in
        correspondence with the reduced proper fractions.
        """
        if self.__symmetry_class_estimate is None:
            order_estimate  = self.symmetry_order_estimate()
            # mode_wraps defines the number of "periods" of abs_fft_xy_rfc to wrap on itself to determine class_signal_v.
            mode_wraps      = 16
            abs_fft_xy_rfc  = np.abs(self.fft_xy_resampled_flow_curve(sample_count=scipy.fftpack.next_fast_len(order_estimate*mode_wraps)))
            # Periodically sum abs_fft_xy_rfc on itself with period given by order_estimate, then divide out by
            # the number of samples summed to get the mean of each slice.
            class_signal_v  = np.zeros((order_estimate,), dtype=abs_fft_xy_rfc.dtype)
            signal_weight_v = np.zeros((order_estimate,), dtype=abs_fft_xy_rfc.dtype)
            # The frequencies produced by this call to scipy.fftpack.fftfreq should already be integral valued,
            # but round anyway just in case, since Python float -> int conversion rounds down.
            mode_v          = np.round(scipy.fftpack.fftfreq(len(abs_fft_xy_rfc), d=1.0/len(abs_fft_xy_rfc))).astype(int)
            for mode,magnitude in zip(mode_v, abs_fft_xy_rfc):
                index = mode%order_estimate
                class_signal_v[index] += magnitude
                signal_weight_v[index] += 1.0
            assert np.all(signal_weight_v > 0)
            class_signal_v /= signal_weight_v
            # The symmetry class estimate is the mode having the highest magnitude.
            self.__symmetry_class_estimate = np.argmax(class_signal_v)
            self.__symmetry_class_signal_v = class_signal_v
            assert self.__symmetry_class_estimate is not None
            assert self.__symmetry_class_signal_v is not None
        return self.__symmetry_class_estimate

    def symmetry_class_signal_v (self):
        if self.__symmetry_class_signal_v is None:
            self.symmetry_class_estimate()
            assert self.__symmetry_class_estimate is not None
            assert self.__symmetry_class_signal_v is not None
        return self.__symmetry_class_signal_v

    def t_v (self):
        if self.__t_v is None:
            self.flow_curve()
            assert self.__t_v is not None
        return self.__t_v

    def Q_v (self):
        if self.__Q_v is None:
            # Let s denote squared distance function s(t) := 1/2 |qp_0 - flow_of_qp_0(t))|^2
            #self.__Q_v = 0.5 * np.sum(np.square(self.flow_curve() - self.qp_0), axis=-1)
            self.__Q_v = vorpy.apply_along_axes(lambda x:0.5*np.sum(np.square(x)), (-2,-1), (self.flow_curve() - self.qp_0,), output_axis_v=(), func_output_shape=())
            assert self.__Q_v is not None
        return self.__Q_v

    def t_min (self):
        if self.__t_min is None:
            self.compute_t_min_and_objective()
            assert self.__t_min is not None
        return self.__t_min

    def objective (self):
        if self.__objective is None:
            self.compute_t_min_and_objective()
            assert self.__objective is not None
        return self.__objective

    def Q_global_min_index (self):
        if self.__Q_global_min_index is None:
            self.compute_t_min_and_objective()
        return self.__Q_global_min_index

    def __call__ (self):
        return self.objective()

    def compute_t_min_and_objective (self):
        t_v                                     = self.t_v()
        Q_v                                     = self.Q_v()

        local_min_index_v                       = [i for i in range(1,len(Q_v)-1) if Q_v[i-1] > Q_v[i] and Q_v[i] < Q_v[i+1]]
        Q_local_min_v                           = [Q_v[i] for i in local_min_index_v]
        try:
            Q_local_min_min_index               = np.argmin(Q_local_min_v)
            self.__Q_global_min_index           = _Q_global_min_index = local_min_index_v[Q_local_min_min_index]
            if True:
                # Fit a quadratic function to the 3 points centered on the argmin in order to have
                # sub-sample accuracy when calculating the objective function value.
                assert 1 <= _Q_global_min_index < len(Q_v)-1
                s                               = slice(_Q_global_min_index-1, _Q_global_min_index+2)
                self.__t_min,self.__objective   = util.quadratic_min_time_parameterized(t_v[s], Q_v[s])
                # Some tests show this discrepancy to be on the order of 1.0e-9
                #print('self.__objective - Q_v[_Q_global_min_index] = {0}'.format(self.__objective - Q_v[_Q_global_min_index]))
            else:
                self.__t_min                    = t_v[_Q_global_min_index]
                self.__objective                = Q_v[_Q_global_min_index]
        except ValueError:
            # If there was no local min, then declare the objective function value to be NaN
            self.__Q_global_min_index           = None
            self.__t_min                        = np.nan
            self.__objective                    = np.nan

    def data_to_pickle (self):
        # First, ensure everything is computed.
        pickle_data = {
            't_v':self.t_v(),
            'initial_preimage':self.preimage_qp_0,
            'initial':self.qp_0,
            'qp_v':self.flow_curve(),
            'Q_v':self.Q_v(),
            'Q_global_min_index':self.Q_global_min_index(),
            't_min':self.t_min(),
            'obj':self.objective(),
            'symmetry_order_estimate':self.symmetry_order_estimate(),
            'symmetry_class_estimate':self.symmetry_class_estimate(),
            'symmetry_class_signal_v':self.symmetry_class_signal_v(),
        }
        # Not sure if there's a guarantee as to the order the above dict elements are computed, so
        # ensure that self.flow_curve() has been called before assigning the flow_curve_was_salvaged
        # attribute in the dict.
        pickle_data['flow_curve_was_salvaged'] = self.flow_curve_was_salvaged,
        return pickle_data

def evaluate_shooting_method_objective (dynamics_context, qp_0, t_max, t_delta, disable_salvage=False):
    """A utility function for constructing a ShootingMethodObjective instance and evaluating it."""
    #print('evaluate_shooting_method_objective; trying qp_0 = {0}'.format(qp_0))
    smo = ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_0, t_max=t_max, t_delta=t_delta, disable_salvage=disable_salvage)
    objective = smo.objective()
    return objective

