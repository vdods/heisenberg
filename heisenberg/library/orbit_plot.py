from . import heisenberg_dynamics_context
import matplotlib.pyplot as plt
import numpy as np
import vorpy

# Short names identifying the different kinds of plots that can be created
valid_quantity_to_plot_v = [
    'x,y',
    't,z',
    'error(H)',
    'error(J)',
    'sqd',
    'objective',
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
}

class OrbitPlot:
    def __init__ (self, *, curve_description_v, quantity_to_plot_v):
        assert len(frozenset(curve_description_v)) == len(curve_description_v), 'must specify unique values in curve_description_v'
        assert frozenset(quantity_to_plot_v).issubset(frozenset(valid_quantity_to_plot_v)), 'specified invalid elements of quantity_to_plot_v: {0}'.format(','.join(frozenset(quantity_to_plot_v).difference(frozenset(valid_quantity_to_plot_v))))

        self.curve_description_v  = curve_description_v
        self.quantity_to_plot_v  = quantity_to_plot_v

        row_count               = len(curve_description_v)
        col_count               = len(quantity_to_plot_v)

        self.fig,self.axis_vv   = plt.subplots(row_count, col_count, squeeze=False, figsize=(15*col_count,15*row_count))

    def plot_curve_quantity (self, *, axis, curve_description, quantity_to_plot, smo, objective_history_v=None, disable_plot_decoration=False):
        """
        Plot a particular quantity from the curve.  This is called by plot_curve, and you shouldn't
        normally need to call this, unless you want very specific control over the plotting.
        """
        assert quantity_to_plot in valid_quantity_to_plot_v, 'quantity_to_plot (which is {0}) was not found in valid_quantity_to_plot_v (which is {1})'.format(quantity_to_plot, valid_quantity_to_plot_v)

        title = '{0} {1}'.format(curve_description, valid_plot_to_include_title_d[quantity_to_plot])

        if quantity_to_plot == 'x,y':
            flow_curve = smo.flow_curve()
            Q_global_min_index = smo.Q_global_min_index()

            axis.plot(0, 0, 'o', color='black')
            axis.plot(flow_curve[:,0,0], flow_curve[:,0,1])
            axis.plot(flow_curve[0,0,0], flow_curve[0,0,1], 'o', color='green', alpha=0.5)
            # TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
            if Q_global_min_index is not None and not disable_plot_decoration:
                axis.plot(flow_curve[Q_global_min_index,0,0], flow_curve[Q_global_min_index,0,1], 'o', color='red', alpha=0.5)
            axis.set_aspect('equal')
            # TODO: option to disable axes
        elif quantity_to_plot == 't,z':
            title += ' (red line indicates t_min, which is {0:.10e})'.format(smo.t_min())
            axis.axhline(0, color='black')
            axis.plot(smo.t_v(), smo.flow_curve()[:,0,2])
            axis.axvline(smo.t_min(), color='red')
        elif quantity_to_plot == 'p_x,p_y':
            flow_curve = smo.flow_curve()
            Q_global_min_index = smo.Q_global_min_index()

            axis.plot(0, 0, 'o', color='black')
            axis.plot(flow_curve[:,1,0], flow_curve[:,1,1])
            axis.plot(flow_curve[0,1,0], flow_curve[0,1,1], 'o', color='green', alpha=0.5)
            # TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
            if Q_global_min_index is not None and not disable_plot_decoration:
                axis.plot(flow_curve[Q_global_min_index,1,0], flow_curve[Q_global_min_index,1,1], 'o', color='red', alpha=0.5)
            axis.set_aspect('equal')
            # TODO: option to disable axes
        elif quantity_to_plot == 't,p_z':
            title += ' (red line indicates t_min, which is {0:.10e})'.format(smo.t_min())
            axis.axhline(0, color='black')
            axis.plot(smo.t_v(), smo.flow_curve()[:,1,2])
            axis.axvline(smo.t_min(), color='red')
        elif quantity_to_plot == 'error(H)':
            H_v = vorpy.apply_along_axes(heisenberg_dynamics_context.Numeric.H, (-2,-1), (smo.flow_curve(),), output_axis_v=(), func_output_shape=())
            abs_H_v = np.abs(H_v)
            title += ' (should stay close to 0)\nmax(abs(H)) = {0:.2e}, H(t=0) = {1:e}'.format(np.max(abs_H_v), H_v[0])
            axis.semilogy(smo.t_v(), abs_H_v)
        elif quantity_to_plot == 'error(J)':
            J_v = vorpy.apply_along_axes(heisenberg_dynamics_context.Numeric.J, (-2,-1), (smo.flow_curve(),), output_axis_v=(), func_output_shape=())
            J_0 = J_v[0]
            J_v -= J_0
            abs_J_minus_J_0 = np.abs(J_v)
            title += ' (should stay close to 0)\nJ(t=0) = {0}; max(abs(J - J(t=0))) = {1:.2e}'.format(J_0, np.max(abs_J_minus_J_0))
            axis.semilogy(smo.t_v(), abs_J_minus_J_0)
        elif quantity_to_plot == 'sqd':
            title += '\nt_min = {0:.10e}, min sqd = {1:.17e}'.format(smo.t_min(), smo.objective())
            axis.semilogy(smo.t_v(), smo.Q_v())
            axis.axvline(smo.t_min(), color='red')
        elif quantity_to_plot == 'objective':
            assert objective_history_v is not None, 'must specify objective_history_v in order to plot {0}'.format(quantity_to_plot)
            title += '\nminimum objective value = {0:.17e}'.format(np.min(objective_history_v))
            axis.semilogy(objective_history_v)
        else:
            assert False, 'this should never happen'

        if disable_plot_decoration:
            axis.set_axis_off()
        else:
            axis.set_title(title)

    def plot_curve (self, *, curve_description, smo, objective_history_v=None, disable_plot_decoration=False):
        """
        curve_description specifies which row to plot to (indexed by curve_description_v specified in the constructor).
        The quantity_to_plot_v value specified in the constructor indicates which quantities from smo to plot.
        """

        assert curve_description in self.curve_description_v, 'curve_description (which is {0}) was not found in curve_description_v (which is {1})'.format(curve_description, curve_description_v)

        row_index = self.curve_description_v.index(curve_description)
        axis_v = self.axis_vv[row_index]

        print('plot_curve; curve_description = "{0}"'.format(curve_description))
        assert len(axis_v) == len(self.quantity_to_plot_v)
        for axis,quantity_to_plot in zip(axis_v, self.quantity_to_plot_v):
            plot_it = True
            if quantity_to_plot == 'objective' and objective_history_v is None:
                plot_it = False
            if plot_it:
                print('plotting quantity "{0}" for curve "{1}"'.format(quantity_to_plot, curve_description))
                self.plot_curve_quantity(axis=axis, curve_description=curve_description, quantity_to_plot=quantity_to_plot, smo=smo, objective_history_v=objective_history_v, disable_plot_decoration=disable_plot_decoration)
            else:
                print('NOT plotting quantity "{0}" for curve "{1}"'.format(quantity_to_plot, curve_description))

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

