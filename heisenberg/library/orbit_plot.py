from . import heisenberg_dynamics_context
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from . import util
import vorpy

# Short names identifying the different kinds of plots that can be created
valid_quantity_to_plot_v = [
    'x,y',
    't,z',
    'p_x,p_y',
    't,p_z',
    'error(H)',
    'error(J)',
    'sqd',
    'objective',
    'resampled-x,y',
    'fft-x,y',
    'class-signal',
    'resampled-t,z',
    'fft-t,z',
]

valid_plot_to_include_title_d = {
    'x,y'           : 'curve position in (x,y) only',
    't,z'           : 'curve position in z, time-parameterized',
    'p_x,p_y'       : 'curve momentum in (x,y) only',
    't,p_z'         : 'curve momentum in z, time-parameterized',
    'error(H)'      : 'abs(H)',
    'error(J)'      : 'abs(J - J(t=0))',
    'sqd'           : 'squared distance from initial',
    'objective'     : 'objective function history',
    'resampled-x,y' : 'testing x,y portion of resampled_flow_curve',
    'fft-x,y'       : 'testing fft_xy_resampled_flow_curve',
    'class-signal'  : 'testing class-signal',
    'resampled-t,z' : 'testing t,z portion of resampled_flow_curve',
    'fft-t,z'       : 'testing fft_z_resampled_flow_curve',
}

terse_valid_plot_to_include_title_d = {
    'x,y'           : '(x(t),y(t))',
    't,z'           : '(t,z(t))',
    'p_x,p_y'       : '(p_x(t),p_y(t))',
    't,p_z'         : '(t,p_z(t)',
    'error(H)'      : 'abs(H)',
    'error(J)'      : 'abs(J - J(t=0))',
    'sqd'           : 'sqr dist from IC',
    'objective'     : 'obj func hist',
    'resampled-x,y' : 'x,y portion of resampled_flow_curve',
    'fft-x,y'       : 'fft_xy_resampled_flow_curve',
    'class-signal'  : 'class-signal',
    'resampled-t,z' : 't,z portion of resampled_flow_curve',
    'fft-t,z'       : 'fft_z_resampled_flow_curve',
}

default_quantities_to_plot = 'x,y;t,z;error(H);error(J);sqd;class-signal;objective'
default_quantity_to_plot_v = default_quantities_to_plot.split(';')

class OrbitPlot:
    def __init__ (self, *, curve_description_v, quantity_to_plot_v, size):
        assert type(quantity_to_plot_v) == list
        assert len(frozenset(curve_description_v)) == len(curve_description_v), 'must specify unique values in curve_description_v'
        assert frozenset(quantity_to_plot_v).issubset(frozenset(valid_quantity_to_plot_v)), 'specified invalid elements of quantity_to_plot_v: {0}'.format(','.join(frozenset(quantity_to_plot_v).difference(frozenset(valid_quantity_to_plot_v))))

        self.curve_description_v  = curve_description_v
        self.quantity_to_plot_v  = quantity_to_plot_v

        row_count               = len(curve_description_v)
        col_count               = len(quantity_to_plot_v)

        size                    = size
        self.fig,self.axis_vv   = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    def plot_curve_quantity (self, *, axis, curve_description, quantity_to_plot, smo, objective_history_v=None, cut_off_curve_tail=False, disable_plot_decoration=False, use_terse_titles=False):
        """
        Plot a particular quantity from the curve.  This is called by plot_curve, and you shouldn't
        normally need to call this, unless you want very specific control over the plotting.
        """
        assert quantity_to_plot in valid_quantity_to_plot_v, 'quantity_to_plot (which is {0}) was not found in valid_quantity_to_plot_v (which is {1})'.format(quantity_to_plot, valid_quantity_to_plot_v)

        if use_terse_titles:
            title = terse_valid_plot_to_include_title_d[quantity_to_plot]
        else:
            title = '{0} {1}'.format(curve_description, valid_plot_to_include_title_d[quantity_to_plot])

        # end_t_index is the time-index to plot stuff to.
        end_t_index = len(smo.t_v())
        Q_global_min_index = smo.Q_global_min_index()
        actually_cut_off_curve_tail = cut_off_curve_tail and Q_global_min_index is not None
        if actually_cut_off_curve_tail:
            end_t_index = Q_global_min_index

        if quantity_to_plot == 'x,y':
            flow_curve = smo.flow_curve()

            axis.plot(0, 0, 'o', color='black')
            axis.plot(flow_curve[:end_t_index,0,0], flow_curve[:end_t_index,0,1], color='black')
            axis.plot(flow_curve[0,0,0], flow_curve[0,0,1], 'o', color='green', alpha=0.5)
            # TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
            if Q_global_min_index is not None and not disable_plot_decoration:
                axis.plot(flow_curve[Q_global_min_index,0,0], flow_curve[Q_global_min_index,0,1], 'o', color='red', alpha=0.5)
            axis.set_aspect('equal')
        elif quantity_to_plot == 't,z':
            if not use_terse_titles:
                if not actually_cut_off_curve_tail:
                    title += ' (red line indicates t_min, which is {0:.10e})'.format(smo.t_min())
                else:
                    title += ' (t_min is {0:.10e})'.format(smo.t_min())
            axis.axhline(0, color='black')
            axis.plot(smo.t_v()[:end_t_index], smo.flow_curve()[:end_t_index,0,2], color='black')
            if not actually_cut_off_curve_tail:
                axis.axvline(smo.t_min(), color='red')
        elif quantity_to_plot == 'p_x,p_y':
            flow_curve = smo.flow_curve()
            Q_global_min_index = smo.Q_global_min_index()

            axis.plot(0, 0, 'o', color='black')
            axis.plot(flow_curve[:end_t_index,1,0], flow_curve[:end_t_index,1,1], color='black')
            axis.plot(flow_curve[0,1,0], flow_curve[0,1,1], 'o', color='green', alpha=0.5)
            # TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
            if not actually_cut_off_curve_tail and not disable_plot_decoration:
                axis.plot(flow_curve[Q_global_min_index,1,0], flow_curve[Q_global_min_index,1,1], 'o', color='red', alpha=0.5)
            axis.set_aspect('equal')
        elif quantity_to_plot == 't,p_z':
            if not use_terse_titles:
                if not actually_cut_off_curve_tail:
                    title += ' (red line indicates t_min, which is {0:.10e})'.format(smo.t_min())
                else:
                    title += ' (t_min is {0:.10e})'.format(smo.t_min())
            axis.axhline(0, color='black')
            axis.plot(smo.t_v()[:end_t_index], smo.flow_curve()[:end_t_index,1,2], color='black')
            if not actually_cut_off_curve_tail:
                axis.axvline(smo.t_min(), color='red')
        elif quantity_to_plot == 'error(H)':
            H_v = vorpy.apply_along_axes(heisenberg_dynamics_context.Numeric.H, (-2,-1), (smo.flow_curve(),), output_axis_v=(), func_output_shape=())
            abs_H_v = np.abs(H_v)
            if not use_terse_titles:
                title += ' (should stay close to 0)\nmax(abs(H)) = {0:.2e}, H(t=0) = {1:e}'.format(np.max(abs_H_v), H_v[0])
            axis.semilogy(smo.t_v()[:end_t_index], abs_H_v[:end_t_index], color='black')
        elif quantity_to_plot == 'error(J)':
            J_v = vorpy.apply_along_axes(heisenberg_dynamics_context.Numeric.J, (-2,-1), (smo.flow_curve(),), output_axis_v=(), func_output_shape=())
            J_0 = J_v[0]
            J_v -= J_0
            abs_J_minus_J_0 = np.abs(J_v)
            if not use_terse_titles:
                title += ' (should stay close to 0)\nJ(t=0) = {0}; max(abs(J - J(t=0))) = {1:.2e}'.format(J_0, np.max(abs_J_minus_J_0))
            axis.semilogy(smo.t_v()[:end_t_index], abs_J_minus_J_0[:end_t_index], color='black')
            if not actually_cut_off_curve_tail:
                axis.axvline(smo.t_min(), color='red')
        elif quantity_to_plot == 'sqd':
            if not use_terse_titles:
                title += '\nt_min = {0:.10e}, min sqd = {1:.17e}'.format(smo.t_min(), smo.objective())
            axis.semilogy(smo.t_v()[:end_t_index], smo.Q_v()[:end_t_index], color='black')
            if not actually_cut_off_curve_tail:
                axis.axvline(smo.t_min(), color='red')
        elif quantity_to_plot == 'objective':
            assert objective_history_v is not None, 'must specify objective_history_v in order to plot {0}'.format(quantity_to_plot)
            if not use_terse_titles:
                title += '\nminimum objective value = {0:.17e}'.format(np.min(objective_history_v))
            axis.semilogy(objective_history_v, color='black')
        elif quantity_to_plot == 'resampled-x,y':
            #sample_count    = 1024
            sample_count    = 128
            rfc             = smo.resampled_flow_curve(sample_count=sample_count)
            axis.plot(0, 0, 'o', color='black')
            axis.scatter(rfc[:end_t_index,0,0], rfc[:end_t_index,0,1], color='black', s=1)
            axis.plot(rfc[0,0,0], rfc[0,0,1], 'o', color='green', alpha=0.5)
            axis.set_aspect('equal')
        elif quantity_to_plot == 'fft-x,y':
            sample_count    = 1024
            fft_xy_rfc      = smo.fft_xy_resampled_flow_curve(sample_count=sample_count)
            fft_freq        = np.round(scipy.fftpack.fftfreq(sample_count, d=1.0/sample_count)).astype(int)
            axis.semilogy(fft_freq, np.abs(fft_xy_rfc), 'o', color='black')
        elif quantity_to_plot == 'class-signal':
            symmetry_order_estimate = smo.symmetry_order_estimate()
            symmetry_class_estimate = smo.symmetry_class_estimate()
            symmetry_class_signal_v = smo.symmetry_class_signal_v()
            axis.axvline(symmetry_class_estimate, color='green')
            axis.semilogy(symmetry_class_signal_v, 'o', color='black')
            if not use_terse_titles:
                title += '\nclass_signal; class estimate = {0}, order estimate: {1}'.format(symmetry_class_estimate, symmetry_order_estimate)
        elif quantity_to_plot == 'resampled-t,z':
            sample_count    = 128
            rt              = smo.resampled_time(sample_count=sample_count)
            rfc             = smo.resampled_flow_curve(sample_count=sample_count)
            axis.plot(rt, rfc[:end_t_index,0,2], 'o', color='black')
        elif quantity_to_plot == 'fft-t,z':
            #n = 8
            #sample_count = 4*n
            sample_count = 1024 # TODO: Need to pick this more intelligently
            z_rfc = smo.resampled_flow_curve(sample_count=sample_count)[:,0,2]
            fft_z_rfc = smo.fft_z_resampled_flow_curve(sample_count=sample_count)

            fft_freq = scipy.fftpack.fftfreq(len(fft_z_rfc))
            #print('fft_freq:')
            #print('    \n'.join(str(f) for f in fft_freq))
            #axis.plot(fft_freq, np.real(fft_z_rfc), 'o', color='green')
            #axis.plot(fft_freq, np.imag(fft_z_rfc), 'o', color='blue')
            #axis.semilogy(fft_freq, np.abs(fft_z_rfc), 'o')
            abs_fft_z_rfc = np.abs(fft_z_rfc)
            # Calculate the strongest mode -- put into descending order -- strongest first
            strength_sorted_mode_index = np.argsort(abs_fft_z_rfc)[::-1]
            assert len(strength_sorted_mode_index) == len(fft_freq)
            # Make a copy in which the upper half of the indices negative.
            signed_strength_sorted_mode_index = np.copy(strength_sorted_mode_index)
            for i in range(len(signed_strength_sorted_mode_index)):
                if signed_strength_sorted_mode_index[i] >= sample_count//2:
                    signed_strength_sorted_mode_index[i] -= sample_count
            axis.axvline(strength_sorted_mode_index[0], color='green')
            axis.axvline(strength_sorted_mode_index[1], color='blue')
            axis.semilogy(abs_fft_z_rfc, 'o', color='black')
            if not use_terse_titles:
                title += '\nstrongest and second strongest modes: {0} and {1}'.format(signed_strength_sorted_mode_index[0], signed_strength_sorted_mode_index[1])
        else:
            assert False, 'this should never happen'

        if disable_plot_decoration:
            axis.set_axis_off()
        else:
            axis.set_title(title)

    def plot_curve (self, *, curve_description, smo, objective_history_v=None, cut_off_curve_tail=False, disable_plot_decoration=False, use_terse_titles=False):
        """
        curve_description specifies which row to plot to (indexed by curve_description_v specified in the constructor).
        The quantity_to_plot_v value specified in the constructor indicates which quantities from smo to plot.
        """

        assert curve_description in self.curve_description_v, 'curve_description (which is {0}) was not found in curve_description_v (which is {1})'.format(curve_description, curve_description_v)

        row_index = self.curve_description_v.index(curve_description)
        axis_v = self.axis_vv[row_index]

        #print('plot_curve; curve_description = "{0}"'.format(curve_description))
        assert len(axis_v) == len(self.quantity_to_plot_v)
        for axis,quantity_to_plot in zip(axis_v, self.quantity_to_plot_v):
            plot_it = True
            if quantity_to_plot == 'objective' and objective_history_v is None:
                plot_it = False
            if plot_it:
                #print('plotting quantity "{0}" for curve "{1}"'.format(quantity_to_plot, curve_description))
                self.plot_curve_quantity(axis=axis, curve_description=curve_description, quantity_to_plot=quantity_to_plot, smo=smo, objective_history_v=objective_history_v, cut_off_curve_tail=cut_off_curve_tail, disable_plot_decoration=disable_plot_decoration, use_terse_titles=use_terse_titles)
            else:
                #print('NOT plotting quantity "{0}" for curve "{1}"'.format(quantity_to_plot, curve_description))
                pass

    def savefig_and_clear (self, *, filename):
        self.fig.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        print('wrote to file "{0}"'.format(filename))
        # VERY important to do this -- otherwise your memory will slowly fill up!
        # Not sure which one is actually sufficient -- apparently none of them are, YAY!
        plt.clf()
        plt.close(self.fig)
        plt.close('all')
        del self.fig
        del self.axis_vv

