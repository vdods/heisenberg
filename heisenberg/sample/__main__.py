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
    '--sample-count',
    dest='sample_count',
    default=1000,
    type='int',
    help='Specify the number of samples to generate.  Default is 1000.'
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

options,args = op.parse_argv_and_validate()
if options is None:
    sys.exit(-1)

rng = np.random.RandomState(options.seed)
heisenberg.sample.sample(dynamics_context, options, rng=rng)
