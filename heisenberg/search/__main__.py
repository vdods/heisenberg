import heisenberg.library.heisenberg_dynamics_context
import heisenberg.option_parser
import heisenberg.search
import matplotlib
import numpy as np
import sys

# https://github.com/matplotlib/matplotlib/issues/5907 says this should fix "Exceeded cell block limit" problems
matplotlib.rcParams['agg.path.chunksize'] = 10000

dynamics_context = heisenberg.library.heisenberg_dynamics_context.Numeric()

op = heisenberg.option_parser.OptionParser(module=heisenberg.search)
# Add the subprogram-specific options here.
op.add_option(
    '--optimization-iterations',
    dest='optimization_iterations',
    default=1000,
    type='int',
    help='Specifies the number of iterations to run the optimization for (if applicable).  Default is 1000.'
)
op.add_option(
    '--abortive-threshold',
    dest='abortive_threshold',
    default=0.1,
    type='float',
    help='Sets the threshold below which a candidate curve\'s objective function will qualify it for running through the optimizer to attempt to close it.'
)
op.add_option(
    '--output-dir',
    dest='output_dir',
    default='search-results',
    help='Sets the directory into which the search will generate plot images and data.  Default is "search-results".'
)
op.add_option(
    '--abortive-subdir',
    dest='abortive_subdir',
    default=None,
    help='If specified, sets the subdirectory of the output dir into which the search will generate plot images and data for initial conditions whose objective function value was not below the abortive threshold.'
)

options,args = op.parse_argv_and_validate(dynamics_context)
if options is None:
    sys.exit(-1)

rng = np.random.RandomState(options.seed)
heisenberg.search.search(dynamics_context, options, rng=rng)
