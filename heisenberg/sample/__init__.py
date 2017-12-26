import concurrent.futures
import heisenberg.library.shooting_method_objective
import heisenberg.library.util
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
    dynamics_context, initial_preimage, dt, max_time, embedding_dimension, embedding_solution_sheet_index = args[0], args[1], args[2], args[3], args[4], args[5]
    qp_0 = dynamics_context.embedding(N=embedding_dimension, sheet_index=embedding_solution_sheet_index)(initial_preimage)

    try:
        # Use disable_salvage=True to avoid clogging up the place.
        smo = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, preimage_qp_0=initial_preimage, qp_0=qp_0, t_max=max_time, t_delta=dt, disable_salvage=True)

        flow_curve = smo.flow_curve()
        objective = smo.objective()
        t_min = smo.t_min()
        abs_H_v = np.abs(vorpy.apply_along_axes(dynamics_context.H, (-2,-1), (flow_curve,)))

        J_v = vorpy.apply_along_axes(dynamics_context.J, (-2,-1), (flow_curve,))
        J_0 = J_v[0]
        J_v -= J_0
        abs_J_minus_J_0 = np.abs(J_v)

        sample_result = make_sample_result(
            initial=initial_preimage,
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

    if options.worker_chunksize <= 0:
        print('--worker-chunksize option, if present, must specify a positive integer.')
        sys.exit(-1)

    if options.max_workers is not None and options.max_workers <= 0:
        print('--max-workers option, if present, must specify a positive integer.')
        sys.exit(-1)

    heisenberg.util.ensure_dir_exists(options.samples_dir)

    # Ensure that the requested embedding is already vorpy.symbolic.cache_lambdified, because otherwise
    # the worker processes somehow crash, not being able to generate it themselves.
    dynamics_context.embedding(N=options.embedding_dimension, sheet_index=options.embedding_solution_sheet_index)

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
        'full_commandline': heisenberg.util.reconstruct_full_commandline(executable=sys.executable, argv=sys.argv),
        'options': vars(options), # vars ensures it's a dict, and not a stupid optparse.Values object.
        'sample_v': sample_result_v,
    }

    print('saving results...')
    maybe_seed_string = 'seed:{0}.'.format(options.seed) if options.sampling_type == 'random' else ''
    base_filename = os.path.join(options.samples_dir, 'sample_v.{0}count:{1}'.format(maybe_seed_string, len(sample_result_v)))
    vorpy.pickle.try_to_pickle(data=data, pickle_filename=base_filename+'.pickle', log_out=sys.stdout)
    # Also create a human-readable summary of the pickle data.
    heisenberg.util.write_human_readable_summary(data=data, filename=base_filename+'.summary')

    if options.embedding_dimension == 1 and options.sampling_type == 'ordered':
        print('finding minima of the objective function and classifying the curves.')

        p_y_v = []
        objective_v = []
        t_min_v = []

        # Create lists of the relevant sample_result values.
        for sample_result in sample_result_v:
            # See make_sample_result for which element is which.

            # sample_result[0] is initial preimage, which in the case of --embedding-dimension=1, is the initial p_y (aka p_theta) value.
            assert np.shape(sample_result[0]) == (1,) # Should be a 1-vector.
            p_y_v.append(sample_result[0][0])
            # sample_result[4] is the objective function value for that initial condition.
            assert np.shape(sample_result[4]) == tuple() # Should be a scalar.
            objective_v.append(sample_result[4])
            # sample_result[5] is t_min, which is the time at which the curve most nearly closed up on itself.
            assert np.shape(sample_result[5]) == tuple() # Should be a scalar.
            t_min_v.append(sample_result[5])

        # Turn those lists into np.array objects.
        p_y_v                   = np.array(p_y_v)
        objective_v             = np.array(objective_v)
        t_min_v                 = np.array(t_min_v)

        # Compute all local minima of the objective function
        local_min_index_v       = [i for i in range(1,len(objective_v)-1) if objective_v[i-1] > objective_v[i] and objective_v[i] < objective_v[i+1]]
        # Use exp quadratic fit to compute time of local mins at sub-sample accuracy -- use
        # exp_quadratic_min_time_parameterized so that the computed min is nonnegative.
        local_min_v             = []
        for local_min_index in local_min_index_v:
            # For each local min, we only care about its discrete "neighborhood".
            s                   = slice(local_min_index-1, local_min_index+2)
            # Just take the "local" slice of p_y_v and objective_v
            p_y_local_v         = p_y_v[s]
            objective_local_v   = objective_v[s]
            p_y_min,objective   = heisenberg.library.util.exp_quadratic_min_time_parameterized(p_y_local_v, objective_local_v)
            assert p_y_local_v[0] < p_y_min < p_y_local_v[-1], 'p_y_min is outside the neighborhood of the local min -- this should be impossible'

            t_min_local_v       = t_min_v[s]
            #print('p_y_local_v = {0}, t_min_local_v = {1}, p_y_min = {2}'.format(p_y_local_v, t_min_local_v, p_y_min))
            period              = np.interp(p_y_min, p_y_local_v, t_min_local_v)

            local_min_v.append((p_y_min, objective, period))

        # Go through each local min, compute the curve, classify it, and plot it if indicated by the --for-each-1d-minimum=classify-and-plot option.
        print('Curve classifications (symmetry class and order values are estimates):')
        print('Format is "class:order <=> { initial_p_y=<value>, objective=<value>, period=<value>, dt=<value> }')

        try:
            plot_commands_filename = base_filename+'.plot_commands'
            plot_commands_file = open(plot_commands_filename, 'w')
        except IOError as e:
            print('was not able to open file "{0}" for writing; not writing plot commands.'.format(plot_commands_filename))
            plot_commands_file = None

        for local_min in local_min_v:
            # TODO: Make this parallelized.
            initial_p_y = np.array([local_min[0]])
            qp_0 = dynamics_context.embedding(N=options.embedding_dimension, sheet_index=options.embedding_solution_sheet_index)(initial_p_y)

            objective = local_min[1]

            period = local_min[2]
            # Go a little bit further than the period so there's at least some overlap to work with.
            max_time = 1.1*period

            # Create the object that will actually compute everything.
            smo = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, preimage_qp_0=initial_p_y, qp_0=qp_0, t_max=max_time, t_delta=options.dt, disable_salvage=True)

            # Print the classification along with enough info to reconstruct the curve fully.
            classification_string = 'order: {0}, class: {1}, symmetry type: {1}/{0} <=> {{ initial_p_y={2}, objective={3}, period={4}, dt={5} }}'.format(
                smo.symmetry_order_estimate(),
                smo.symmetry_class_estimate(),
                initial_p_y,
                objective,
                period,
                options.dt
            )
            print(classification_string)

            # Print an example command to use heisenberg.plot to plot that curve.
            plot_command = 'python3 -m heisenberg.plot --dt={0} --max-time={1} --initial-preimage=[{2}] --embedding-dimension=1 --embedding-solution-sheet-index={3} --output-dir="{4}" --quantities-to-plot="x,y;t,z;error(H);error(J);sqd;class-signal;objective" --plot-type=pdf # {5}'.format(
                options.dt,
                max_time,
                initial_p_y[0],
                options.embedding_solution_sheet_index,
                options.samples_dir+'/plots',
                classification_string
            )
            print('Command to plot:')
            print(plot_command)
            plot_commands_file.write(plot_command)
            plot_commands_file.write('\n')
