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
    help='Sets the threshold below which a candidate curve\'s objective function will qualify it for running through the optimizer to attempt to close it.  The default is 0.1 -- changing this has not really produced better results, so it is recommended to keep the default.'
)
op.add_option(
    '--output-dir',
    dest='output_dir',
    default='search-results',
    help='Sets the directory into which the search will generate plot images and data.  Default is "search-results".'
)
op.add_option(
    '--disable-abortive-output',
    dest='disable_abortive_output',
    action='store_true',
    default=False,
    help='If specified, disables the saving of data upon discovery of an "abortive" search result.  Default behavior is to save data to the "abortive" subdirectory of the directory specified by --output-dir.'
)
op.add_option(
    '--exit-after-number-of-successes',
    dest='exit_after_number_of_successes',
    default=0,
    type='int',
    help='Specifies the number of "successes" after which to stop searching.  A "success" is defined by having found a curve that came close enough to closing up to then perform an optimization to refine it into a fully-closed curve.  Specifying any non-positive number indicates that there should be no limit.  The default value is 0.'
)
op.add_option(
    '--exit-after-number-of-tries',
    dest='exit_after_number_of_tries',
    default=0,
    type='int',
    help='Specifies the number of "tries" after which to stop searching.  A "try" is defined by generating a random initial condition and checking if it comes close enough to closing up to be worth performing an optimization to refine it into a fully-closed curve; a "try" counts regardless of it does or does not come close enough to closing up.  Specifying any non-positive number indicates that there should be no limit.  The default value is 0.'
)

options,args = op.parse_argv_and_validate()
if options is None:
    sys.exit(-1)

if options.embedding_dimension != 2 or options.embedding_solution_sheet_index != 1:
    print('Error: The heisenberg.search subprogram requires --embedding-dimension=2 and --embedding-solution-sheet-index=1 (this restriction is artificial and may be removed later).')
    sys.exit(-1)

rng = np.random.RandomState(options.seed)
heisenberg.search.search(dynamics_context, options, rng=rng)
