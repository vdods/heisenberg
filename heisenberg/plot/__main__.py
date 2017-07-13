import heisenberg.library.heisenberg_dynamics_context
import heisenberg.library.orbit_plot
import heisenberg.option_parser
import heisenberg.plot
import heisenberg.util
import matplotlib
import numpy as np
import sys

# https://github.com/matplotlib/matplotlib/issues/5907 says this should fix "Exceeded cell block limit" problems
matplotlib.rcParams['agg.path.chunksize'] = 10000

dynamics_context = heisenberg.library.heisenberg_dynamics_context.Numeric()

op = heisenberg.option_parser.OptionParser(module=heisenberg.plot)
# Add the subprogram-specific options here.
op.add_option(
    '--k-fold-initial',
    dest='k',
    type='int',
    help='Specifies that the given value, call it k, should be used in a particular form of initial condition intended to produce a k-fold symmetric orbit -- experimental.'
)
op.add_option(
    '--initial-1preimage',
    dest='initial_1preimage',
    type='string',
    help='Specifies the preimage of the initial conditions with respect to the [p_y] |-> [[1,0,0],[0,p_y,p_z]] embedding.  Should have the form [p_y], where p_y is a floating point literal.'
)
op.add_option(
    '--initial-2preimage',
    dest='initial_2preimage',
    type='string',
    help='Specifies the preimage of the initial conditions with respect to the [p_x,p_y] |-> [[1,0,0],[p_x,p_y,p_z]] embedding.  Should have the form [p_x,p_y], where each of p_x,p_y are floating point literals.'
)
op.add_option(
    '--initial-3preimage',
    dest='initial_3preimage',
    type='string',
    help='Specifies the preimage of the initial conditions with respect to the [x,p_x,p_y] |-> [[x,0,0],[p_x,p_y,p_z]] embedding.  Should have the form [x,p_x,p_y], where each of x,y,z are floating point literals.'
)
op.add_option(
    '--initial-5preimage',
    dest='initial_5preimage',
    type='string',
    help='Specifies the preimage of the initial conditions with respect to the [x,y,z,p_x,p_y] |-> [[x,y,z],[p_x,p_y,p_z]] embedding.  Should have the form [x,y,z,p_x,p_y], where each of x,y,z are floating point literals.'
)
op.add_option(
    '--initial',
    dest='initial',
    type='string',
    help='Specifies the initial conditions [x,y,z,p_x,p_y,p_z], where each of x,y,z,p_x,p_y,p_z are floating point literals.'
)
op.add_option(
    '--optimization-iterations',
    dest='optimization_iterations',
    default=1000,
    type='int',
    help='Specifies the number of iterations to run the optimization for (if applicable).  Default is 1000.'
)
op.add_option(
    '--optimize-initial',
    dest='optimize_initial',
    action='store_true',
    default=False,
    help='Indicates that the specified initial condition (via whichever of the --initial... options) should be used as the starting point for an optimization to attempt to close the orbit.  Default value is False.'
)
op.add_option(
    '--output-dir',
    dest='output_dir',
    default='.',
    help='Specifies the directory to write plot images and data files to.  Default is current directory.'
)
op.add_option(
    '--disable-plot-initial',
    dest='disable_plot_initial',
    action='store_true',
    default=False,
    help='Disables plotting the initial curve; only has effect if --optimize-initial is specified.'
)
op.add_option(
    '--disable-plot-decoration',
    dest='disable_plot_decoration',
    action='store_true',
    default=False,
    help='Disables plotting certain non-essential labels and decoration.  Default behavior is to plot those things.'
)

default_quantity_to_plot_v = 'x,y;t,z;error(H);error(J);sqd;objective'
op.add_option(
    '--quantities-to-plot',
    dest='quantities_to_plot',
    type='str',
    default=default_quantity_to_plot_v,
    help='Specifies which quantities to include in the plot.  Should be a semicolon-separated string, without spaces, with tokens selected from the following options: {0}.  Default is {1}'.format(';'.join(heisenberg.library.orbit_plot.valid_quantity_to_plot_v), default_quantity_to_plot_v)
)

options,args = op.parse_argv_and_validate()
if options is None:
    sys.exit(-1)

num_initial_conditions_specified = sum([
    options.initial_1preimage is not None,
    options.initial_2preimage is not None,
    options.initial_3preimage is not None,
    options.initial_5preimage is not None,
    options.initial is not None,
    options.k is not None
])
if num_initial_conditions_specified != 1:
    print('Some initial condition option must be specified; --k-fold-initial, --initial-2preimage, --initial-1preimage, --initial-3preimage, --initial-5preimage, --initial.  However, {0} of those were specified.'.format(num_initial_conditions_specified))
    op.print_help()
    sys.exit(-1)

# Validate subprogram-specific options here.

# Attempt to parse initial conditions.  Upon success, the attribute options.qp_0 should exist.
if options.k is not None:
    options.initial_k_fold = np.array([1.0, 0.0, 0.25*np.sqrt(options.k**4 * np.pi**2 * 0.0625 - 1.0), 0.0, 1.0/options.k])
    options.qp_0 = dynamics_context.embedding(5)(options.initial_k_fold)
elif options.initial_1preimage is not None:
    # TODO: Refactor this checking to avoid code duplication
    try:
        options.initial_1preimage = heisenberg.util.csv_as_ndarray(heisenberg.util.pop_brackets_off_of(options.initial_1preimage), float)
        expected_shape = (1,)
        if options.initial_1preimage.shape != expected_shape:
            raise ValueError('--initial-1preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_1preimage.shape, expected_shape))
        options.qp_0 = dynamics_context.embedding(1)(options.initial_1preimage)
    except ValueError as e:
        print('error parsing --initial-1preimage value: {0}'.format(str(e)))
        op.print_help()
        sys.exit(-1)
elif options.initial_2preimage is not None:
    try:
        options.initial_2preimage = heisenberg.util.csv_as_ndarray(heisenberg.util.pop_brackets_off_of(options.initial_2preimage), float)
        expected_shape = (2,)
        if options.initial_2preimage.shape != expected_shape:
            raise ValueError('--initial-2preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_2preimage.shape, expected_shape))
        options.qp_0 = dynamics_context.embedding(2)(options.initial_2preimage)
    except ValueError as e:
        print('error parsing --initial-2preimage value: {0}'.format(str(e)))
        op.print_help()
        sys.exit(-1)
elif options.initial_3preimage is not None:
    try:
        options.initial_3preimage = heisenberg.util.csv_as_ndarray(heisenberg.util.pop_brackets_off_of(options.initial_3preimage), float)
        expected_shape = (3,)
        if options.initial_3preimage.shape != expected_shape:
            raise ValueError('--initial-3preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_3preimage.shape, expected_shape))
        options.qp_0 = dynamics_context.embedding(3)(options.initial_3preimage)
    except ValueError as e:
        print('error parsing --initial-3preimage value: {0}'.format(str(e)))
        op.print_help()
        sys.exit(-1)
elif options.initial_5preimage is not None:
    try:
        options.initial_5preimage = heisenberg.util.csv_as_ndarray(heisenberg.util.pop_brackets_off_of(options.initial_5preimage), float)
        expected_shape = (5,)
        if options.initial_5preimage.shape != expected_shape:
            raise ValueError('--initial-5preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_5preimage.shape, expected_shape))
        options.qp_0 = dynamics_context.embedding(5)(options.initial_5preimage)
    except ValueError as e:
        print('error parsing --initial-5preimage value: {0}'.format(str(e)))
        op.print_help()
        sys.exit(-1)
elif options.initial is not None:
    try:
        options.initial = heisenberg.util.csv_as_ndarray(heisenberg.util.pop_brackets_off_of(options.initial), float)
        expected_shape = (6,)
        if options.initial.shape != expected_shape:
            raise ValueError('--initial value had the wrong number of components (got {0} but expected {1}).'.format(options.initial.shape, expected_shape))
        options.qp_0 = options.initial.reshape(2,3)
    except ValueError as e:
        print('error parsing --initial value: {0}'.format(str(e)))
        op.print_help()
        sys.exit(-1)
else:
    assert False, 'this should never happen because of the check with num_initial_conditions_specified'

# Parse options.quantities_to_plot
options.quantity_to_plot_v = options.quantities_to_plot.split(';')
assert all(plot_to_include in heisenberg.library.orbit_plot.valid_quantity_to_plot_v for plot_to_include in options.quantity_to_plot_v)

rng = np.random.RandomState(options.seed)
heisenberg.plot.plot(dynamics_context, options, rng=rng)
