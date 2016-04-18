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
    import warnings

    warnings.filterwarnings('ignore', module='matplotlib')

    if not os.path.exists('opt'):
        os.mkdir('opt')

    def plot_stuff (pc, fdp, fd_fc, fig, axis_row, name, objective_function_value_v=None, t=None):
        color_v = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
        qv_v = pc.curve(fd_fc)
        H_v = [pc.H(pc.time_domain_parameters_at_t(t_,fdp)) for t_ in xrange(pc.fourier_curve_parameterization.T)]
        H_v.append(pc.H(pc.time_domain_parameters_at_t(0,fdp)))
        for d in xrange(pc.D):
            a = axis_row[d]
            a.axvline(0, color='black', alpha=0.5)
            a.axhline(0, color='black', alpha=0.5)
            a.set_title('{0}\n{1}'.format(['position','velocity'][d], name))

            n = 0

            a.plot(qv_v[:,0,d], qv_v[:,1,d], color=color_v[n%len(color_v)])
            a.plot([qv_v[-1,0,d],qv_v[0,0,d]], [qv_v[-1,1,d],qv_v[0,1,d]], color=color_v[n%len(color_v)])
            if t == None:
                a.scatter(qv_v[:,0,d], qv_v[:,1,d], color=color_v[n%len(color_v)], s=2)
            else:
                a.scatter(qv_v[t,0,d], qv_v[t,1,d], color=color_v[n%len(color_v)], s=2)

            a.plot([0], [0], 'ro', color='black')
            a.set_aspect(1.0)

            a = axis_row[pc.D+d]
            a.set_title('{0}\n{1}'.format(['position','velocity'][d], name))

            # for time in pc.fourier_curve_parameterization.half_open_time_interval:
            #     a.axvline(time, color='black', alpha=0.5, lw=0.5)
            a.plot(pc.fourier_curve_parameterization.half_open_time_interval, qv_v[:,0,d], color='blue')
            a.plot(pc.fourier_curve_parameterization.closed_time_interval[-2:], [qv_v[-1,0,d],qv_v[0,0,d]], color='blue')
            a.plot(pc.fourier_curve_parameterization.half_open_time_interval, qv_v[:,1,d], color='green')
            a.plot(pc.fourier_curve_parameterization.closed_time_interval[-2:], [qv_v[-1,1,d],qv_v[0,1,d]], color='green')

        a = axis_row[-3]
        a.set_title('Hamiltonian (max - min = {0})'.format(np.max(H_v) - np.min(H_v)))
        # for time in pc.fourier_curve_parameterization.half_open_time_interval:
        #     a.axvline(time, color='black', alpha=0.5, lw=0.5)
        a.plot(pc.fourier_curve_parameterization.closed_time_interval, H_v)
        if t != None:
            a.axvline(pc.fourier_curve_parameterization.half_open_time_interval[t])

        a = axis_row[-2]
        a.set_title('Fourier coefficients (log abs)')
        a.semilogy(pc.fourier_curve_parameterization.frequencies, np.linalg.norm(fd_fc, axis=1))

        a = axis_row[-1]
        if objective_function_value_v != None:
            objective_function_value_min = np.min(objective_function_value_v)
            a.set_title('objective function for each iteration\nplot value is delta from min value {0}'.format(objective_function_value_min))
            objective_function_deltas = objective_function_value_v - objective_function_value_min
            if np.any(objective_function_deltas > 0.0):
                a.semilogy(objective_function_deltas)
            a.set_xlim(0, len(objective_function_value_v))

    pickle_filename = 'library/kepler.pickle'
    with open(pickle_filename, 'rb') as f:
        import pickle
        d = pickle.load(f)
        frequencies                 = d['frequencies']
        period                      = d['period']
        fourier_coefficient_tensor  = d['fourier_coefficient_tensor']

    print 'creating KeplerProblemContext...'
    sample_count = 400
    pc = kepler.KeplerProblemContext(np.linspace(0.0, period, sample_count+1), frequencies, use_constraint=False)
    # pc = kepler.KeplerProblemContext(T=30, F=2, use_constraint=False)
    # pc = NBodyProblemContext(T=120, F=5, use_constraint=False)
    print 'pc.frequency_domain_parameter_count =', pc.frequency_domain_parameter_count

    fdp,fd_fc,fd_g,fd_lm = pc.make_frequency_domain_parameters_and_views()

    if True:
        fd_fc[pc.fourier_curve_parameterization.index_of_frequency(0),0] = 0.6 # constant offset in x
        a = 1.0 # x-radius
        b = 0.7 # y-radius
        fd_fc[pc.fourier_curve_parameterization.index_of_frequency( 1),0] = 0.5*(a+b)
        fd_fc[pc.fourier_curve_parameterization.index_of_frequency(-1),0] = 0.5*(a-b)
    else:
        assert fd_fc.shape == fourier_coefficient_tensor.shape
        fd_fc[:] = fourier_coefficient_tensor

    fd_g[0] = 1.0

    row_count = 10
    col_count = 7
    overview_fig,overview_axes = plt.subplots(row_count, col_count, squeeze=False, figsize=(8*col_count,8*row_count))
    plot_stuff(pc, fdp, fd_fc, overview_fig, overview_axes[0], 'pre-optimization')

    bsgc = gradient_descent.BatchedStochasticGradientComputer(pc.DLambda, pc.fourier_curve_parameterization.T)#, pc.frequency_domain_parameter_count)
    # bsgc = gradient_descent.BatchedStochasticGradientComputer(pc.DLambda, pc.fourier_curve_parameterization.T, 20)#, pc.frequency_domain_parameter_count)
    # gd = gradient_descent.GradientDescent(pc.Lambda, bsgc.eval_gradient_on_current_batch, fdp, np.logspace(log_learning_rate_range[0], log_learning_rate_range[1], learning_rate_count))
    # gd = gradient_descent.GradientDescent(pc.Lambda, bsgc.eval_gradient_on_current_batch, fdp, np.logspace(-3.5,-6,9))
    # gd = gradient_descent.GradientDescent(pc.Lambda, bsgc.eval_gradient_on_current_batch, fdp, np.array([1.0e-6]))
    gd = gradient_descent.BlindGradientDescent(pc.Lambda, bsgc.eval_gradient_on_current_batch, fdp, 1.0e-4)

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
        row_index = 1
        max_iteration_count = (row_count-1)*1000
        print_progress_message(0, fdp, fd_lm)
        for iteration_index in xrange(max_iteration_count):
            fdp[:] = gd.compute_next_step(fdp, (iteration_index+1)%10==0)
            bsgc.go_to_next_batch()
            # if (iteration_index+1)%100 == 0:
            #     # Reparameterize the Fourier curve by arclength
            #     pc.fourier_curve_parameterization = pc.fourier_curve_parameterization.arclength_reparameterization_2(fd_fc, pc.fourier_curve_parameterization.derivatives==0)
            make_report = (iteration_index+1)%(max_iteration_count//(row_count-1)) == 0
            if make_report:
                print_progress_message(iteration_index, fdp, fd_lm)
                plot_stuff(pc, fdp, fd_fc, overview_fig, overview_axes[row_index], 'after {0} iterations'.format(iteration_index), gd.obj_history_v)
                row_index += 1
    except KeyboardInterrupt:
        # Allow this so we can interrupt the optimization if we get impatient.
        print 'caught KeyboardInterrupt; plotting results so far.'

    solution_delta = fd_fc - fourier_coefficient_tensor
    max_abs_solution_delta = np.max(np.abs(solution_delta))
    print 'max_abs_solution_delta:', max_abs_solution_delta

    # axis = overview_axes[0][-1]
    # axis.set_title('learning rate')
    # if np.any(gd.best_learning_rate_history_v - np.min(gd.best_learning_rate_history_v) > 0.0):
    #     axis.semilogy(gd.best_learning_rate_history_v)
    #     axis.set_ylim(np.min(gd.learning_rate_v), np.max(gd.learning_rate_v))

    overview_fig.tight_layout()
    filename = 'opt/overview.png'
    plt.savefig(filename)
    print 'wrote "{0}"'.format(filename)
    plt.close(overview_fig)
    del overview_fig

    sys.exit(0)
