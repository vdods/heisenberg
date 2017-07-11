from . import heisenberg_dynamics_context
import matplotlib.pyplot as plt
import numpy as np
import vorpy

class OrbitPlot:
    def __init__ (self, *, row_count, extra_col_count):
        row_count = row_count
        col_count = 5+extra_col_count
        self.fig,self.axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(15*col_count,15*row_count))

    def plot_curve (self, *, curve_description, axis_v, smo):
        flow_curve = smo.flow_curve()
        H_v = vorpy.apply_along_axes(heisenberg_dynamics_context.Numeric.H, (-2,-1), (flow_curve,), output_axis_v=(), func_output_shape=())
        abs_H_v = np.abs(H_v)
        Q_global_min_index = smo.Q_global_min_index()

        axis_index = 0

        axis = axis_v[axis_index]
        axis.set_title('{0} curve xy-position'.format(curve_description))
        axis.plot(0, 0, 'o', color='black')
        axis.plot(flow_curve[:,0,0], flow_curve[:,0,1])
        axis.plot(flow_curve[0,0,0], flow_curve[0,0,1], 'o', color='green', alpha=0.5)
        # TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
        if Q_global_min_index is not None:
            axis.plot(flow_curve[Q_global_min_index,0,0], flow_curve[Q_global_min_index,0,1], 'o', color='red', alpha=0.5)
        axis.set_aspect('equal')
        axis_index += 1

        axis = axis_v[axis_index]
        axis.set_title('{0} curve z-position (green line indicates t_min, which is {1:.10e})'.format(curve_description, smo.t_min()))
        axis.axhline(0, color='black')
        axis.plot(smo.t_v(), flow_curve[:,0,2])
        axis.axvline(smo.t_min(), color='green')
        axis_index += 1

        #axis = axis_v[axis_index]
        #axis.set_title('{0} curve xy-momentum'.format(curve_description))
        #axis.plot(flow_curve[:,1,0], flow_curve[:,1,1])
        #axis.plot(flow_curve[0,1,0], flow_curve[0,1,1], 'o', color='green', alpha=0.5)
        ## TODO: Plot the interpolated position/momentum (based on smo.t_min() instead of Q_global_min_index)
        #if Q_global_min_index is not None:
        #    axis.plot(flow_curve[Q_global_min_index,1,0], flow_curve[Q_global_min_index,1,1], 'o', color='red', alpha=0.5)
        #axis.set_aspect('equal')
        #axis_index += 1

        #axis = axis_v[axis_index]
        #axis.set_title('{0} curve z-momentum'.format(curve_description))
        #axis.axhline(0, color='black')
        #axis.plot(smo.t_v(), flow_curve[:,1,2])
        #axis.axvline(smo.t_min(), color='green')
        #axis_index += 1

        axis = axis_v[axis_index]
        axis.set_title('abs(H) (should stay close to 0)\nmax(abs(H)) = {0:.2e}, H_0 = {1:e}'.format(np.max(abs_H_v), H_v[0]))
        axis.semilogy(smo.t_v(), abs_H_v)
        axis_index += 1

        J_v = vorpy.apply_along_axes(heisenberg_dynamics_context.Numeric.J, (-2,-1), (flow_curve,), output_axis_v=(), func_output_shape=())
        J_0 = J_v[0]
        J_v -= J_0
        abs_J_minus_J_0 = np.abs(J_v)

        axis = axis_v[axis_index]
        axis.set_title('abs(J - J_0) (should be close to 0)\nJ_0 = {0}; max(abs(J - J_0)) = {1:.2e}'.format(J_0, np.max(abs_J_minus_J_0)))
        axis.semilogy(smo.t_v(), abs_J_minus_J_0)
        axis_index += 1

        axis = axis_v[axis_index]
        axis.set_title('squared distance to initial condition\nt_min = {0}, min sqd = {1:.17e}'.format(smo.t_min(), smo.objective()))
        axis.semilogy(smo.t_v(), smo.Q_v())
        axis.axvline(smo.t_min(), color='green')
        axis_index += 1

    def plot_and_clear (self, *, filename):
        self.fig.tight_layout()
        plt.savefig(filename)
        print('wrote to file "{0}"'.format(filename))
        # VERY important to do this -- otherwise your memory will slowly fill up!
        # Not sure which one is actually sufficient -- apparently none of them are, YAY!
        plt.clf()
        plt.close(self.fig)
        plt.close('all')
        del self.fig
        del self.axis_vv

