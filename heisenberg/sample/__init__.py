import heisenberg.library.shooting_method_objective
import heisenberg.util
import numpy as np
import os
import sys
import vorpy
import vorpy.pickle
import vorpy.symplectic_integration.exceptions

## TODO: Come up with less generic name.
#Sample = collections.namedtuple('Sample', ['initial2', 'qp_0', 'dt', 'max_time', 'objective', 't_min', 'max_abs_H', 'max_abs_J_minus_J_0', 'flow_curve_was_salvaged'])
def make_sample (*, initial2, qp_0, dt, max_time, objective, t_min, max_abs_H, max_abs_J_minus_J_0, flow_curve_was_salvaged):
    return (initial2, qp_0, dt, max_time, objective, t_min, max_abs_H, max_abs_J_minus_J_0, flow_curve_was_salvaged)

def sample (dynamics_context, options, *, rng):
    """
    Program to search for orbits in the 2D embedding space:
    -   Generate random points in the domain

            -sqrt(4/pi) <= p_x <= sqrt(4/pi)
                     -C <= p_y <= C

        for some arbitrary positive constant C, say 2.  Due to a discrete symmetry in the system
        (reflection), p_y can be taken to be nonnegative.  Thus the domain can be

            -sqrt(4/pi) <= p_x <= sqrt(4/pi)
                      0 <= p_y <= C

    -   For each of these, compute the embedding qp_0 into phase space and use that as the initial
        condition for the flow curve.  Use some fixed dt and max_time.
    -   For each flow curve, compute the following values:
        -   The objective function value (which could be NaN if the curve didn't go back toward itself)
        -   The t_min value (which for a closed curve would be its period)
        -   The upper bound on abs(H), or more detailed stats (mean and variance, or a histogram)
        -   The upper bound on abs(J - J_0), or more detailed stats (mean and variance, or a histogram)
        -   If it is [close to being] closed, then the order of its radial symmetry
        -   The deviation of its z(t) function from a sine wave (for closed, this sometimes but not
            always appears to be a sine wave, but this hasn't been checked rigorously).
    -   Visualize these two functions to determine if they are continuous and/or have other structure
        to them.  Perhaps there are regions which are continuous surrounding zeros the objective function.
    """

    if options.seed is None:
        print('--seed must be specified.')
        sys.exit(-1)

    if not os.path.exists(options.samples_dir):
        # TODO: Create dirs recursively if it's a nested dir
        os.mkdir(options.samples_dir)

    sample_v = []
    try:
        while True:
            initial2 = heisenberg.util.random_embedding2_point(rng)
            qp_0 = dynamics_context.embedding2(initial2)
            dt = options.dt
            max_time = options.max_time

            try:
                smo = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_0, t_max=max_time, t_delta=dt)

                flow_curve = smo.flow_curve()
                objective = smo.objective()
                t_min = smo.t_min()
                abs_H_v = np.abs(vorpy.apply_along_axes(dynamics_context.H, (-2,-1), (flow_curve,)))

                J_v = vorpy.apply_along_axes(dynamics_context.J, (-2,-1), (flow_curve,))
                J_0 = J_v[0]
                J_v -= J_0
                abs_J_minus_J_0 = np.abs(J_v)

                sample = make_sample(
                    initial2=initial2,
                    qp_0=qp_0,
                    dt=dt,
                    max_time=max_time,
                    objective=objective,
                    t_min=t_min,
                    max_abs_H=np.max(abs_H_v),
                    max_abs_J_minus_J_0=np.max(abs_J_minus_J_0),
                    flow_curve_was_salvaged=smo.flow_curve_was_salvaged
                )
                print('recording sample {0}'.format(sample))
            except vorpy.symplectic_integration.exceptions.SalvagedResultException as e:
                print('caught exception "{0}" -- storing and continuing'.format(e))
                sample = make_sample(
                    initial2=initial2,
                    qp_0=qp_0,
                    dt=dt,
                    max_time=max_time,
                    objective=np.nan,
                    t_min=np.nan,
                    max_abs_H=np.nan,
                    max_abs_J_minus_J_0=np.nan,
                    flow_curve_was_salvaged=True
                )

            sample_v.append(sample)
    except (Exception,KeyboardInterrupt) as e:
        print('caught exception "{0}" -- saving results and exiting.'.format(e))
        filename = os.path.join(options.samples_dir, 'sample_v.seed:{0}.count:{1}.pickle'.format(options.seed, len(sample_v)))
        vorpy.pickle.try_to_pickle(data=sample_v, pickle_filename=filename, log_out=sys.stdout)

