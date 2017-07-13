import heisenberg.library.monte_carlo
import heisenberg.library.orbit_plot
import heisenberg.library.shooting_method_objective
import heisenberg.util
import os

subprogram_description = 'Plots an integral curve of the system using given initial condition, optionally running an optimization method to find a nearby curve that closes back up on itself.'

def plot (dynamics_context, options, *, rng):
    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir, exist_ok=True)

    # Plot given curve
    qp_0 = options.qp_0
    smo_0 = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=options.qp_0, t_max=options.max_time, t_delta=options.dt)
    print('smo_0.objective() = {0}'.format(smo_0.objective()))

    # Construct curve_description_v
    curve_description_v = []
    if not options.disable_plot_initial:
        curve_description_v += ['initial curve']
    if options.optimize_initial:
        curve_description_v += ['optimized curve']

    op = heisenberg.library.orbit_plot.OrbitPlot(curve_description_v=curve_description_v, quantity_to_plot_v=options.quantity_to_plot_v)

    if options.optimize_initial:
        if options.initial_1preimage is not None:
            X_0 = options.initial_1preimage
            embedding = dynamics_context.embedding(1)
        elif options.initial_2preimage is not None:
            X_0 = options.initial_2preimage
            embedding = dynamics_context.embedding(2)
        elif options.initial_3preimage is not None:
            X_0 = options.initial_3preimage
            embedding = dynamics_context.embedding(3)
        elif options.initial_5preimage is not None:
            X_0 = options.initial_5preimage
            embedding = dynamics_context.embedding(5)
        elif options.initial is not None:
            X_0 = options.qp_0
            embedding = None
        else:
            assert options.initial_k_fold is not None
            X_0 = options.initial_k_fold
            embedding = dynamics_context.embedding(5)

        disable_salvage = True
        optimizer = heisenberg.library.monte_carlo.MonteCarlo(
            obj=lambda qp_0:heisenberg.library.shooting_method_objective.evaluate_shooting_method_objective(dynamics_context, qp_0, options.max_time, options.dt, disable_salvage),
            initial_parameters=X_0,
            inner_radius=1.0e-12,
            outer_radius=1.0e-1,
            rng_seed=options.seed, # TODO: Make a single RNG that's used everywhere in the program.
            embedding=embedding
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
        smo_opt = heisenberg.library.shooting_method_objective.ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_opt, t_max=options.max_time, t_delta=options.dt)

        print('qp_opt = {0}'.format(qp_opt))
        if embedding is not None:
            print('qp_opt embedding preimage; X_0 = {0}'.format(optimizer.parameter_history_v[-1]))

        op.plot_curve(curve_description='optimized curve', smo=smo_opt, objective_history_v=optimizer.obj_history_v, disable_plot_decoration=options.disable_plot_decoration)

        #axis = op.axis_vv[-1][-1]
        #axis.set_title('objective function history')
        #axis.semilogy(optimizer.obj_history_v)

        qp = qp_opt
        smo = smo_opt
    else:
        qp = qp_0
        smo = smo_0

    base_filename = os.path.join(
        options.output_dir,
        heisenberg.util.construct_base_filename(
            obj=smo.objective(),
            t_delta=options.dt,
            t_max=options.max_time,
            initial_condition=qp,
            t_min=smo.t_min(),
            k=options.k
        )
    )

    if not options.disable_plot_initial:
        print('plotting initial curve')
        op.plot_curve(curve_description='initial curve', smo=smo_0, disable_plot_decoration=options.disable_plot_decoration)
    else:
        print('NOT plotting initial curve')

    op.savefig_and_clear(filename=base_filename+'.'+options.plot_type)
    smo.pickle(base_filename+'.pickle')
