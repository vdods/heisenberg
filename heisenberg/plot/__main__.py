import ast
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
    '--initial-preimage',
    dest='initial_preimage',
    type='string',
    help='Specifies the preimage of the initial conditions with respect to the embedding map specified by the --embedding-dimension and --embedding-solution-sheet-index option values.  Should have the form [x_1,...,x_n], where n is the embedding dimension and x_i is a floating point literal for each i.'
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

options,args = op.parse_argv_and_validate()
if options is None:
    sys.exit(-1)

num_initial_conditions_specified = sum([
    options.initial_preimage is not None,
    options.initial is not None,
])
if num_initial_conditions_specified != 1:
    print('Some initial condition option must be specified; --initial-preimage, --initial.  However, {0} of those were specified.'.format(num_initial_conditions_specified))
    op.print_help()
    sys.exit(-1)

# Validate subprogram-specific options here.

# Attempt to parse initial conditions.  Upon success, the attribute options.qp_0 should exist.
if options.initial_preimage is not None:
    try:
        options.initial_preimage = np.array(ast.literal_eval(options.initial_preimage))
        expected_shape = (options.embedding_dimension,)
        if options.initial_preimage.shape != expected_shape:
            raise ValueError('--initial-preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_preimage.shape, expected_shape))
        options.qp_0 = dynamics_context.embedding(N=options.embedding_dimension, sheet_index=options.embedding_solution_sheet_index)(options.initial_preimage)
    except Exception as e:
        print('error parsing --initial-preimage value; error was {0}'.format(e))
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

rng = np.random.RandomState(options.seed)
heisenberg.plot.plot(dynamics_context, options, rng=rng)
