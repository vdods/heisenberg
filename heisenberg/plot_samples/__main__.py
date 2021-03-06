import heisenberg.library.heisenberg_dynamics_context
import heisenberg.option_parser
import heisenberg.plot_samples
import matplotlib
import numpy as np
import sys

# https://github.com/matplotlib/matplotlib/issues/5907 says this should fix "Exceeded cell block limit" problems
matplotlib.rcParams['agg.path.chunksize'] = 10000

dynamics_context = heisenberg.library.heisenberg_dynamics_context.Numeric()

op = heisenberg.option_parser.OptionParser(module=heisenberg.plot_samples)
# Add the subprogram-specific options here.
op.add_option(
    '--samples-dir',
    dest='samples_dir',
    default='.',
    type='str',
    help='Specify the directory that sample_v.*.pickle files will be written to.'
)
op.add_option(
    '--use-white-background',
    dest='use_white_background',
    default=False,
    action='store_true',
    help='Specify that the plotting should be done on a white background.'
)

options,args = op.parse_argv_and_validate()
if options is None:
    sys.exit(-1)

rng = np.random.RandomState(options.seed)
heisenberg.plot_samples.plot_samples(dynamics_context, options, rng=rng)
