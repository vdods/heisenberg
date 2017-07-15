import concurrent.futures
import heisenberg.library.shooting_method_objective
import heisenberg.util
import itertools
import numpy as np
import os
import sys
import traceback
import vorpy
import vorpy.pickle
import vorpy.symplectic_integration.exceptions

subprogram_description = 'Samples a specified parameter space of initial conditions, computing the corresponding integral curves, and computing and storing relevant data about the each curve in a file.  This is intended to sample various functions of the initial condition space, and can be used in later processing; see the heisenberg.plot_samples subprogram.'

## TODO: Come up with less generic name.
#Sample = collections.namedtuple('Sample', ['initial', 'qp_0', 'dt', 'max_time', 'objective', 't_min', 'max_abs_H', 'max_abs_J_minus_J_0', 'flow_curve_was_salvaged'])
def make_sample_result (*, initial, qp_0, dt, max_time, objective, t_min, max_abs_H, max_abs_J_minus_J_0, flow_curve_was_salvaged):
    return (initial, qp_0, dt, max_time, objective, t_min, max_abs_H, max_abs_J_minus_J_0, flow_curve_was_salvaged)

def worker (args):
    assert len(args) == 6, 'passed wrong number of arguments to worker function'
    dynamics_context, initial, dt, max_time, embedding_dimension, embedding_solution_sheet_index = args[0], args[1], args[2], args[3], args[4], args[5]
    qp_0 = dynamics_context.embedding(N=embedding_dimension, sheet_index=embedding_solution_sheet_index)(initial)

    try:
        # Use disable_salvage=True to avoid clogging up the place.
        smo = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_0, t_max=max_time, t_delta=dt, disable_salvage=True)

        flow_curve = smo.flow_curve()
        objective = smo.objective()
        t_min = smo.t_min()
        abs_H_v = np.abs(vorpy.apply_along_axes(dynamics_context.H, (-2,-1), (flow_curve,)))

        J_v = vorpy.apply_along_axes(dynamics_context.J, (-2,-1), (flow_curve,))
        J_0 = J_v[0]
        J_v -= J_0
        abs_J_minus_J_0 = np.abs(J_v)

        sample_result = make_sample_result(
            initial=initial,
            qp_0=qp_0,
            dt=dt,
            max_time=max_time,
            objective=objective,
            t_min=t_min,
            max_abs_H=np.max(abs_H_v),
            max_abs_J_minus_J_0=np.max(abs_J_minus_J_0),
            flow_curve_was_salvaged=smo.flow_curve_was_salvaged
        )
        #print('recording sample {0}'.format(sample_result))
    except Exception as e:
        print('caught exception "{0}" -- storing and continuing'.format(e))
        print('stack:')
        ex_type,ex,tb = sys.exc_info()
        traceback.print_tb(tb)
        sample_result = make_sample_result(
            initial=initial,
            qp_0=qp_0,
            dt=dt,
            max_time=max_time,
            objective=np.nan,
            t_min=np.nan,
            max_abs_H=np.nan,
            max_abs_J_minus_J_0=np.nan,
            flow_curve_was_salvaged=True # Change this to "exception occurred" and store info about exception
        )

    return sample_result

def sample (dynamics_context, options, *, rng):
    """
    Program to search for orbits in the 2D embedding space:
    -   Generate random points in the domain

            -sqrt(1/(4*pi)) <= p_x <= sqrt(1/(4*pi))
                         -C <= p_y <= C

        for some arbitrary positive constant C, say 2.  Due to a discrete symmetry in the system
        (reflection), p_y can be taken to be nonnegative.  Thus the domain can be

            -sqrt(1/(4*pi)) <= p_x <= sqrt(1/(4*pi))
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

    if options.sampling_type == 'random':
        if options.seed is None:
            print('If --sampling-type=random, then --seed must be specified.')
            sys.exit(-1)

    if options.sample_count <= 0:
        print('--sample-count option, if present, must specify a positive integer.')
        sys.exit(-1)

    if options.max_workers <= 0:
        print('--worker-chunksize option, if present, must specify a positive integer.')
        sys.exit(-1)

    if options.max_workers <= 0:
        print('--max-workers option, if present, must specify a positive integer.')
        sys.exit(-1)

    heisenberg.util.ensure_dir_exists(options.samples_dir)

    sample_result_v = []

    # Define sample_generator based on the sampling type.
    if options.sampling_type == 'random':
        def random_sample (rng):
            return np.array([rng.uniform(low=options.sampling_domain_bound_v[axis,0], high=options.sampling_domain_bound_v[axis,1]) for axis in range(options.embedding_dimension)])
        sample_generator = (random_sample(rng) for _ in range(options.sample_count))
    elif options.sampling_type == 'ordered':
        # Define uniform samplings of each axis in the sampling domain.
        sample_vv = [np.linspace(options.sampling_domain_bound_v[axis,0], options.sampling_domain_bound_v[axis,1], options.sample_count_v[axis]) for axis in range(options.embedding_dimension)]
        sample_generator = (np.array(sample) for sample in itertools.product(*sample_vv))
    else:
        assert False, 'this should never happen'

    if options.max_workers is None or options.max_workers > 1:
        # Map the worker function over multiple processes.
        with concurrent.futures.ProcessPoolExecutor(max_workers=options.max_workers) as executor:
            try:
                #sample_counter = range(options.sample_count)
                sample_index = 1
                print('options:', options)
                for sample_result in executor.map(worker, ((dynamics_context, sample, options.dt, options.max_time, options.embedding_dimension, options.embedding_solution_sheet_index) for sample in sample_generator), chunksize=options.worker_chunksize):
                    sample_result_v.append(sample_result)
                    print('**************** saving sample {0} (out of {1}): {2}'.format(sample_index, options.sample_count, sample_result))
                    sample_index += 1
            except (Exception,KeyboardInterrupt) as e:
                print('encountered exception of type {0} during sample; calling executor.shutdown(wait=True), saving results, and exiting.  exception was: {1}'.format(type(e), e))
                print('stack:')
                ex_type,ex,tb = sys.exc_info()
                traceback.print_tb(tb)
                executor.shutdown(wait=True)
    else:
        # Run the worker function in this single process.
        try:
            for sample in sample_generator:
                sample_result = worker((dynamics_context, sample, options.dt, options.max_time, options.embedding_dimension, options.embedding_solution_sheet_index))
                sample_result_v.append(sample_result)
        except (Exception,KeyboardInterrupt) as e:
            print('encountered exception of type {0} during sample; saving results and exiting.  exception was: {1}'.format(type(e), e))
            print('stack:')
            ex_type,ex,tb = sys.exc_info()
            traceback.print_tb(tb)

    # Create the data structure that will be pickled.
    data = {
        'options': options,
        'sample_v': sample_result_v,
    }

    print('saving results, and exiting.')
    maybe_seed_string = 'seed:{0}.'.format(options.seed) if options.sampling_type == 'random' else ''
    filename = os.path.join(options.samples_dir, 'sample_v.{0}count:{1}.pickle'.format(maybe_seed_string, len(sample_result_v)))
    vorpy.pickle.try_to_pickle(data=data, pickle_filename=filename, log_out=sys.stdout)
