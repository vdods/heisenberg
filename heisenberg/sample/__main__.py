import ast
import heisenberg.library.heisenberg_dynamics_context
import heisenberg.option_parser
import heisenberg.sample
import matplotlib
import numpy as np
import sys

# https://github.com/matplotlib/matplotlib/issues/5907 says this should fix "Exceeded cell block limit" problems
matplotlib.rcParams['agg.path.chunksize'] = 10000

dynamics_context = heisenberg.library.heisenberg_dynamics_context.Numeric()

op = heisenberg.option_parser.OptionParser(module=heisenberg.sample)
# Add the subprogram-specific options here.
op.add_option(
    '--samples-dir',
    dest='samples_dir',
    default='.',
    type='str',
    help='Specify the directory that sample_v.*.pickle files will be read from.'
)
op.add_option(
    '--worker-chunksize',
    dest='worker_chunksize',
    default=1,
    type='int',
    help='Specify the chunksize for multiprocess worker delegation.  Default is 1.'
)
op.add_option(
    '--max-workers',
    dest='max_workers',
    default=None,
    type='int',
    help='Optionally specify the max number of processes this program should use concurrently.  Default is the number of processors.'
)
op.add_option(
    '--sampling-domain',
    dest='sampling_domain',
    default=None,
    help='Specifies the rectangular domain over which to sample.  If the number of axes is b+1, then this should be in the format [[lower_0,upper_0],[lower_1,upper_1],...,[lower_b,upper_b]], where [lower_k,upper_k] is the range in the kth axis.'
)
op.add_option(
    '--sampling-type',
    dest='sampling_type',
    choices=['random','ordered'],
    default='random',
    help='Specifies which sampling type to use.  Choices are "random" (draw random samples from the uniform distribution on the sampling domain) and "ordered" (samples form an ordered, uniform mesh on the sampling domain).'
)
op.add_option(
    '--sampling-range',
    dest='sampling_range',
    default=None,
    type='str',
    help='Specify the number of samples to generate, where the required format depends on --sampling-type.  If --sampling-type=random, then this should be a positive integer.  If --sampling-type=ordered and the number of axes is b+1, then this should have the form [n_0,n_1,...,n_b], where n_k is the number of samples along the k axis, and therefore the number of total samples is n_0*n_1*...*n_b.'
)

options,args = op.parse_argv_and_validate()
if options is None:
    sys.exit(-1)

if options.sampling_domain is None:
    print('--sampling-domain must be specified.')
    op.print_help()
    sys.exit(-1)

# Parse options.embedding_dimension
try:
    options.embedding_dimension = int(options.embedding_dimension)
except Exception as e:
    print('error while parsing --embedding-dimension={0}; error was {1}'.format(options.embedding_dimension, e))
    op.print_help()
    sys.exit(-1)


assert options.sampling_type is not None

# Parse options.sampling_domain.
try:
    options.sampling_domain_bound_v = ast.literal_eval(options.sampling_domain)
    options.sampling_domain_bound_v = np.array(options.sampling_domain_bound_v, dtype=float)
    assert len(options.sampling_domain_bound_v.shape) == 2, 'value had the wrong shape; expected 2 axes but got {0}.'.format(len(options.sampling_domain_bound_v.shape))
    assert options.sampling_domain_bound_v.shape[0] == options.embedding_dimension, '--embedding-dimension value disagrees with --sampling-domain value.'
    # TODO: Full validation
except Exception as e:
    print('error while parsing --sampling-domain={0}; error was {1}'.format(options.sampling_domain, e))
    op.print_help()
    sys.exit(-1)

# Parse options.sampling_range.
try:
    if options.sampling_type == 'random':
        options.sample_count = int(options.sampling_range)
    elif options.sampling_type == 'ordered':
        options.sample_count_v = ast.literal_eval(options.sampling_range)
        options.sample_count_v = np.array(options.sample_count_v, dtype=int)
        # TODO: Full validation
        options.sample_count = np.prod(options.sample_count_v)
        assert options.sample_count_v.shape[0] == options.embedding_dimension
    else:
        assert False, 'this should never happen'
except Exception as e:
    print('error while parsing --sampling-range={0}; error was {1}'.format(options.sampling_range, e))
    op.print_help()
    sys.exit(-1)

rng = np.random.RandomState(options.seed)
heisenberg.sample.sample(dynamics_context, options, rng=rng)
