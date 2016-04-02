# NOTE: This changes a/b to produce a floating point approximation of that
# ratio, not the integer quotient.  For integer quotient, use a//b instead.
from __future__ import division

import sys

sys.path.append('/Users/vdods/files/github/heisenberg/library')

import bezier_interpolation
import fourier_parameterization
import gradient_descent
import itertools
import kepler
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special
import symbolic
import sympy
import sys
import time

"""
Notes
-----

There is a problem in optimization when the curve is sampled at points whose spacing
becomes very non-homogeneous.  Then part of the curve is overrepresented in the gradient
descent (as if there are many extra validation points, so there's underfitting), while part
of the curve is underrepresented in the gradient descent (too few validation points, so
there's overfitting).  It seems like occasionally re-sampling the curve so that the points
are spaced more uniformly in configuration space (or probably actually phase space) would
address this problem.  This would involve estimating the density of the sampling along
the curve and redistributing the points evenly along the curve.
"""

if __name__ == '__main__':
    import os

    if not os.path.exists('opt'):
        os.mkdir('opt')

    def plot_stuff (pc, fdp, fd_fc, fig, axis_row, name, objective_function_value_v=None, t=None):
        color_v = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
        qv_v = pc.curve(fd_fc)
        H_v = [pc.H(pc.time_domain_parameters_at_t(t_, fdp)) for t_ in xrange(pc.fourier_curve_parameterization.T)]
        for d in xrange(pc.D):
            a = axis_row[d]
            a.axvline(0, color='black', alpha=0.5)
            a.axhline(0, color='black', alpha=0.5)
            a.set_title('{0}\n{1}'.format(['position','velocity'][d], name))

            n = 0

            a.plot(qv_v[:,0,d], qv_v[:,1,d], color=color_v[n%len(color_v)])
            a.plot([qv_v[-1,0,d],qv_v[0,0,d]], [qv_v[-1,1,d],qv_v[0,1,d]], color=color_v[n%len(color_v)])
            if t == None:
                a.plot(qv_v[:,0,d], qv_v[:,1,d], 'ro', color=color_v[n%len(color_v)])
            else:
                a.plot(qv_v[t,0,d], qv_v[t,1,d], 'ro', color=color_v[n%len(color_v)])

            a.plot([0], [0], 'ro', color='black')
            a.set_aspect(1.0)


            a = axis_row[pc.D+d]
            a.set_title('{0}\n{1}'.format(['position','velocity'][d], name))

            a.plot(pc.fourier_curve_parameterization.half_open_time_interval, qv_v[:,0,d])
            a.plot(pc.fourier_curve_parameterization.half_open_time_interval, qv_v[:,1,d])


        a = axis_row[-3]
        a.set_title('Hamiltonian (max - min = {0})'.format(np.max(H_v) - np.min(H_v)))
        a.plot(pc.fourier_curve_parameterization.half_open_time_interval, H_v)
        if t != None:
            a.axvline(pc.fourier_curve_parameterization.half_open_time_interval[t])

        if len(axis_row) > 3:
            a = axis_row[-2]
            a.set_title('Fourier coefficients (log abs)')
            for c in xrange(2):
                a.semilogy(pc.fourier_curve_parameterization.frequencies, np.abs(fd_fc[:,c]))

        if len(axis_row) > 4:
            a = axis_row[-1]
            if objective_function_value_v != None:
                a.set_title('objective function for each iteration\nplot value is delta from min')
                a.semilogy(objective_function_value_v - np.min(objective_function_value_v))

    pc = kepler.KeplerProblemContext(T=100, F=10, use_constraint=False)
    fdp,fd_fc,fd_g,fd_lm = pc.make_frequency_domain_parameters_and_views()

    fd_fc[pc.fourier_curve_parameterization.index_of_frequency(0),0] = 0.6 # constant offset in x
    fd_fc[pc.fourier_curve_parameterization.index_of_frequency(1),0] = 1.0 # cos for x
    fd_g[0] = 1.0

    # pc = NBodyProblemContext(T=120, F=5, use_constraint=False)
    print 'pc.frequency_domain_parameter_count =', pc.frequency_domain_parameter_count

    overview_fig,overview_axes = plt.subplots(2, 7, squeeze=False, figsize=(40,16))
    plot_stuff(pc, fdp, fd_fc, overview_fig, overview_axes[0], 'pre-optimization')

    bsgc = gradient_descent.BatchedStochasticGradientComputer(pc.DLambda, pc.fourier_curve_parameterization.T)#, pc.frequency_domain_parameter_count)
    # gd = gradient_descent.GradientDescent(pc.Lambda, bsgc.eval_gradient_on_current_batch, fdp, np.logspace(log_learning_rate_range[0], log_learning_rate_range[1], learning_rate_count))
    gd = gradient_descent.GradientDescent(pc.Lambda, bsgc.eval_gradient_on_current_batch, fdp, np.logspace(-3.5,-4.5,9))

    # TODO: Try stuff:
    # - Compute the Hessian of the objective function and use that as the Riemannian metric for the parameter
    #   manifold (this is multivariate Newton's method), so that the convergence will be even faster.
    # - Once the objective function stabilizes, reduce alpha by some factor until it stabilizes again, etc.
    #   Maybe use some sort of filtered variance to compute when to do this.

    def print_progress_message (iteration_index, fdp, fd_lm):
        print 'it {0:5}: RMS of Dobjective_function = {1:.15e}, lm = {2}, objective_function = {3:.15e}, C = {4:.15e}'.format(
            iteration_index+1,
            np.sqrt(np.mean(np.square(pc.DLambda(fdp)))),
            fd_lm,
            pc.Lambda(fdp),
            pc.C(fdp),
        )

    try:
        max_iteration_count = 10000
        print_progress_message(0, fdp, fd_lm)
        for iteration_index in xrange(max_iteration_count):
            fdp[:] = gd.compute_next_step(fdp)
            bsgc.go_to_next_batch()
            if (iteration_index+1)%(max_iteration_count//100) == 0:
                print_progress_message(iteration_index, fdp, fd_lm)
    except KeyboardInterrupt:
        pass # Allow this so we can interrupt the optimization if we get impatient.

    plot_stuff(pc, fdp, fd_fc, overview_fig, overview_axes[1], 'post-optimization', gd.obj_history_v)

    axis = overview_axes[0][-1]
    axis.set_title('learning rate')
    axis.semilogy(gd.best_learning_rate_history_v)
    axis.set_ylim(np.min(gd.learning_rate_v), np.max(gd.learning_rate_v))

    overview_fig.tight_layout()
    filename = 'opt/overview.png'
    plt.savefig(filename)
    print 'wrote "{0}"'.format(filename)
    plt.close(overview_fig)
    del overview_fig

    sys.exit(0)
