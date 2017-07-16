import heisenberg.library.monte_carlo
import heisenberg.library.orbit_plot
import heisenberg.library.shooting_method_objective
import heisenberg.library.util
import heisenberg.util
import numpy as np
import os
import sys
import traceback

subprogram_description = 'Search a specified parameter space for initial conditions for curves.  Once a curve is found that comes close enough to closing back up on itself, an optimization method is used to attempt to alter the initial conditions so that the curve closes back up on itself.  The result is integral curves which approximate closed solutions to within numerical accuracy.'

def search (dynamics_context, options, *, rng):
    output_dir = options.output_dir
    abortive_dir = None if options.disable_abortive_output else os.path.join(options.output_dir, 'abortive')

    heisenberg.util.ensure_dir_exists(output_dir)
    if abortive_dir is not None:
        heisenberg.util.ensure_dir_exists(abortive_dir)

    np.set_printoptions(formatter={'float':heisenberg.library.util.float_formatter})

    def try_random_initial_condition ():
        ##X_0 = rng.randn(*heisenberg.library.heisenberg_dynamics_context.Numeric.initial_condition_preimage().shape)
        #X_0 = rng.randn(2)
        ## NOTE: This somewhat biases the generation of random initial conditions
        ##X_0[0] = np.exp(X_0[0]) # So we never get negative values
        #X_0[1] = np.abs(X_0[1]) # So we only bother pointing upward

        # NOTE: The goal here is to sample uniformly over the domain:
        #     -sqrt(1/(4*pi)) <= p_x <= sqrt(1/(4*pi))
        #                   0 <= p_y <= C
        # for some arbitrary positive bound C, say 2.

        #C = 2.0
        ##epsilon = 1.0e-5
        #epsilon = 0.0
        ## Perturb the bounds for p_x by epsilon away from the actual bound.
        #X_0 = np.array([
            #rng.uniform(-np.sqrt(1/(4*np.pi))+epsilon, np.sqrt(1/(4*np.pi))-epsilon),
            #rng.uniform(0.0, C)
        #])
        X_0 = heisenberg.util.random_embedding2_point(rng)

        #X_0 = np.array([4.53918797113298744e-01,-6.06738228528062038e-04,1.75369725636529949e+00])

        qp_0 = dynamics_context.embedding(N=2, sheet_index=1)(X_0)
        print('randomly generated initial condition preimage: X_0:')
        print(X_0)
        #print('embedding of randomly generated initial condition preimage: qp_0:')
        #print(qp_0)
        t_max = 5.0
        # TODO: Pick a large-ish t_max, then cluster the local mins, and then from the lowest cluster,
        # pick the corresponding to the lowest time value, and then make t_max 15% larger than that.
        while True:
            smo_0 = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_0, t_max=t_max, t_delta=options.dt)
            print('smo_0.objective() = {0:.17e}, smo.t_min() = {1}'.format(smo_0.objective(), smo_0.t_min()))
            if smo_0.objective() < options.abortive_threshold:
                break
            else:
                t_max *= 1.5
                print('curve did not nearly close up -- retrying with higher t_max: {0}'.format(t_max))
            if t_max > options.max_time:
                print('t_max ({0}) was raised too many times, exceeding --max-time value of {1}, before nearly closing up -- aborting'.format(t_max, options.max_time))

                if not options.disable_abortive_output:
                    base_filename = os.path.join(
                        abortive_dir,
                        heisenberg.util.construct_base_filename(
                            symmetry_order_estimate=smo_0.symmetry_order_estimate(),
                            symmetry_class_estimate=smo_0.symmetry_class_estimate(),
                            obj=smo_0.objective(),
                            t_delta=options.dt,
                            t_max=t_max,
                            initial_condition=qp_0,
                            t_min=smo_0.t_min()
                        )
                    )

                    orbit_plot = heisenberg.library.orbit_plot.OrbitPlot(curve_description_v=['initial curve'], quantity_to_plot_v=options.quantity_to_plot_v)
                    orbit_plot.plot_curve(curve_description='initial curve', smo=smo_0, cut_off_curve_tail=options.cut_off_initial_curve_tail, disable_plot_decoration=options.disable_plot_decoration)
                    orbit_plot.savefig_and_clear(filename=base_filename+'.'+options.plot_type)
                    smo_0.pickle(base_filename+'.pickle')

                return
        flow_curve_0 = smo_0.flow_curve()

        optimizer = heisenberg.library.monte_carlo.MonteCarlo(
            obj=lambda qp_0:heisenberg.library.shooting_method_objective.evaluate_shooting_method_objective(dynamics_context, qp_0, t_max, options.dt),
            initial_parameters=X_0,
            inner_radius=options.optimization_annulus_bound_v[0],
            outer_radius=options.optimization_annulus_bound_v[-1],
            rng_seed=options.seed,
            embedding=dynamics_context.embedding(N=2, sheet_index=1)
        )
        try:
            actual_iteration_count = 0
            for i in range(options.optimization_iterations):
                optimizer.compute_next_step()
                actual_iteration_count += 1
                print('i = {0}, obj = {1:.17e}'.format(i, optimizer.obj_history_v[-1]))
        except KeyboardInterrupt:
            print('got KeyboardInterrupt -- halting optimization, but will still plot current results')
        except AssertionError:
            print('got AssertionError -- halting optimization, but will plot last good results')

        qp_opt = optimizer.embedded_parameter_history_v[-1]
        smo_opt = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_opt, t_max=t_max, t_delta=options.dt)
        flow_curve_opt = smo_opt.flow_curve()

        print('qp_opt = {0}'.format(qp_opt))
        print('qp_opt embedding preimage; X_0 = {0}'.format(optimizer.parameter_history_v[-1]))

        base_filename = os.path.join(
            output_dir,
            heisenberg.util.construct_base_filename(
                symmetry_order_estimate=smo_opt.symmetry_order_estimate(),
                symmetry_class_estimate=smo_opt.symmetry_class_estimate(),
                obj=smo_opt.objective(),
                t_delta=options.dt,
                t_max=t_max,
                initial_condition=qp_opt,
                t_min=smo_opt.t_min()
            )
        )

        orbit_plot = heisenberg.library.orbit_plot.OrbitPlot(curve_description_v=['initial curve', 'optimized curve'], quantity_to_plot_v=options.quantity_to_plot_v)

        orbit_plot.plot_curve(curve_description='initial curve', smo=smo_0, cut_off_curve_tail=options.cut_off_initial_curve_tail, disable_plot_decoration=options.disable_plot_decoration)
        orbit_plot.plot_curve(curve_description='optimized curve', smo=smo_opt, objective_history_v=optimizer.obj_history_v, cut_off_curve_tail=options.cut_off_optimized_curve_tail, disable_plot_decoration=options.disable_plot_decoration)

        #axis = orbit_plot.axis_vv[0][-1]
        #axis.set_title('objective function history')
        #axis.semilogy(optimizer.obj_history_v)

        orbit_plot.savefig_and_clear(filename=base_filename+'.'+options.plot_type)
        smo_opt.pickle(base_filename+'.pickle')

    try:
        while True:
            try:
                try_random_initial_condition()
            except Exception as e:
                print('encountered exception of type {0} during try_random_initial_condition; skipping.  exception was: {1}'.format(type(e), e))
                print('stack:')
                ex_type,ex,tb = sys.exc_info()
                traceback.print_tb(tb)
    except KeyboardInterrupt:
        print('got KeyboardInterrupt -- exiting program')
        sys.exit(0)

