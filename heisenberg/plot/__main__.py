import heisenberg.library.heisenberg_dynamics_context
import heisenberg.option_parser
import heisenberg.plot
import matplotlib
import numpy as np
import sys

# https://github.com/matplotlib/matplotlib/issues/5907 says this should fix "Exceeded cell block limit" problems
matplotlib.rcParams['agg.path.chunksize'] = 10000

dynamics_context = heisenberg.library.heisenberg_dynamics_context.Numeric()

op = heisenberg.option_parser.OptionParser(prog_name=__package__)
options,args = op.parse_argv_and_validate(dynamics_context)
if options is None:
    sys.exit(-1)

print('options: {0}'.format(options))
print('args   : {0}'.format(args))

rng = np.random.RandomState(options.seed)
heisenberg.plot.plot(dynamics_context, options, rng=rng)
