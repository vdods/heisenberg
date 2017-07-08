import numpy as np
import optparse

class OptionParser:
    def __init__ (self):
        self.op = optparse.OptionParser()
        self.op.add_option(
            '--optimization-iterations',
            dest='optimization_iterations',
            default=1000,
            type='int',
            help='Specifies the number of iterations to run the optimization for (if applicable).  Default is 1000.'
        )
        self.op.add_option(
            '--dt',
            dest='dt',
            default='0.001',
            type='float',
            help='Specifies the timestep for the curve integration.'
        )
        self.op.add_option(
            '--max-time',
            dest='max_time',
            default='50.0',
            type='float',
            help='Specifies the max time to integrate the curve to.'
        )
        self.op.add_option(
            '--initial-2preimage',
            dest='initial_2preimage',
            type='string',
            help='Specifies the preimage of the initial conditions with respect to the [p_x,p_y] |-> [[x,y,z],[p_x,p_y,p_z]] embedding.  Should have the form [p_x,p_y], where each of p_x,p_y are floating point literals.'
        )
        self.op.add_option(
            '--initial-3preimage',
            dest='initial_3preimage',
            type='string',
            help='Specifies the preimage of the initial conditions with respect to the [x,p_x,p_y] |-> [[x,y,z],[p_x,p_y,p_z]] embedding.  Should have the form [x,p_x,p_y], where each of x,y,z are floating point literals.'
        )
        self.op.add_option(
            '--initial-5preimage',
            dest='initial_5preimage',
            type='string',
            help='Specifies the preimage of the initial conditions with respect to the [x,y,z,p_x,p_y] |-> [[x,y,z],[p_x,p_y,p_z]] embedding.  Should have the form [x,y,z,p_x,p_y], where each of x,y,z are floating point literals.'
        )
        self.op.add_option(
            '--initial',
            dest='initial',
            type='string',
            help='Specifies the initial conditions [x,y,z,p_x,p_y,p_z], where each of x,y,z,p_x,p_y,p_z are floating point literals.'
        )
        self.op.add_option(
            '--seed',
            dest='seed',
            default=666,
            type='int',
            help='Specifies the seed to use for pseudorandom number generation.  Using the same seed should produce the same sequence of random numbers, and therefore provide reproducible program execution.'
        )
        self.op.add_option(
            '--optimize-initial',
            dest='optimize_initial',
            action='store_true',
            default=False,
            help='Indicates that the specified initial condition (via whichever of the --initial... options) should be used as the starting point for an optimization to attempt to close the orbit.  Default value is False.'
        )
        self.op.add_option(
            '--search',
            dest='search',
            action='store_true',
            default=False,
            help='Indicates that random initial conditions should be generated and for each, if a threshold is met, an optimization routine run to attempt to close the orbit.'
        )
        self.op.add_option(
            '--sample',
            dest='sample',
            action='store_true',
            default=False,
            help='Indefinitely generates random points in the 2D initial condition space, integrates the curves with the corresponding initial conditions, then records the various statistics for each of them, saving all stats in a pickle file upon program exit (e.g. via KeyboardInterrupt or other exception).'
        )
        self.op.add_option(
            '--plot',
            dest='plot',
            action='store_true',
            default=False,
            help='Plots the given initial conditions.  If the --optimize-initial option is also present, then the given initial condition will be optimized and that will also be plotted.'
        )
        self.op.add_option(
            '--k-fold-initial',
            dest='k',
            type='int',
            help='Specifies that the given value, call it k, should be used in a particular form of initial condition intended to produce a k-fold symmetric orbit -- experimental.'
        )
        self.op.add_option(
            '--abortive-threshold',
            dest='abortive_threshold',
            default=0.1,
            type='float',
            help='Sets the threshold below which a candidate curve\'s objective function will qualify it for running through the optimizer to attempt to close it.'
        )

    @staticmethod
    def __pop_brackets_off_of (string):
        if len(string) < 2:
            raise ValueError('string (which is "{0}") must be at least 2 chars long'.format(string))
        elif string[0] != '[' or string[-1] != ']':
            raise ValueError('string (which is "{0}") must begin with [ and end with ]'.format(string))
        return string[1:-1]

    @staticmethod
    def __csv_as_ndarray (string, dtype):
        return np.array([dtype(token) for token in string.split(',')])

    def parse_argv_and_validate (self, argv, dynamics_context):
        options,args = self.op.parse_args()

        if options.search:
            require_initial_conditions = False
        elif options.sample:
            require_initial_conditions = False
        elif options.k is not None:
            require_initial_conditions = False
            options.qp_0 = np.array([
                [1.0,             0.0, 0.25*np.sqrt(options.k**4 * np.pi**2 * 0.0625 - 1.0)],
                [0.0, 1.0 / options.k,                                                  0.0]
            ])
        else:
            require_initial_conditions = True

        num_initial_conditions_specified = sum([
            options.initial_2preimage is not None,
            options.initial_3preimage is not None,
            options.initial_5preimage is not None,
            options.initial is not None
        ])
        if require_initial_conditions:
            if num_initial_conditions_specified != 1:
                print('if none of --search, --sample, --k-fold-initial are specified, then you must specify exactly one of --initial-2preimage or --initial-3preimage or --initial-5preimage or --initial, but {0} of those were specified.'.format(num_initial_conditions_specified))
                self.op.print_help()
                return None,None

        if options.dt is None:
            print('required option --dt was not specified.')
            self.op.print_help()
            return None,None

        if options.max_time is None:
            print('required option --max-time was not specified.')
            self.op.print_help()
            return None,None

        if require_initial_conditions:
            # Attempt to parse initial conditions.  Upon success, the attribute options.qp_0 should exist.
            if options.initial_2preimage is not None:
                # TODO: Refactor this checking to avoid code duplication
                try:
                    options.initial_2preimage = OptionParser.__csv_as_ndarray(OptionParser.__pop_brackets_off_of(options.initial_2preimage), float)
                    expected_shape = (2,)
                    if options.initial_2preimage.shape != expected_shape:
                        raise ValueError('--initial-2preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_2preimage.shape, expected_shape))
                    options.qp_0 = dynamics_context.embedding2(options.initial_2preimage)
                except ValueError as e:
                    print('error parsing --initial-2preimage value: {0}'.format(str(e)))
                    self.op.print_help()
                    return None,None
            elif options.initial_3preimage is not None:
                try:
                    options.initial_3preimage = OptionParser.__csv_as_ndarray(OptionParser.__pop_brackets_off_of(options.initial_3preimage), float)
                    expected_shape = (3,)
                    if options.initial_3preimage.shape != expected_shape:
                        raise ValueError('--initial-3preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_3preimage.shape, expected_shape))
                    options.qp_0 = dynamics_context.embedding3(options.initial_3preimage)
                except ValueError as e:
                    print('error parsing --initial-3preimage value: {0}'.format(str(e)))
                    self.op.print_help()
                    return None,None
            elif options.initial_5preimage is not None:
                try:
                    options.initial_5preimage = OptionParser.__csv_as_ndarray(OptionParser.__pop_brackets_off_of(options.initial_5preimage), float)
                    expected_shape = (5,)
                    if options.initial_5preimage.shape != expected_shape:
                        raise ValueError('--initial-5preimage value had the wrong number of components (got {0} but expected {1}).'.format(options.initial_5preimage.shape, expected_shape))
                    options.qp_0 = dynamics_context.embedding5(options.initial_5preimage)
                except ValueError as e:
                    print('error parsing --initial-5preimage value: {0}'.format(str(e)))
                    self.op.print_help()
                    return None,None
            elif options.initial is not None:
                try:
                    options.initial = OptionParser.__csv_as_ndarray(OptionParser.__pop_brackets_off_of(options.initial), float)
                    expected_shape = (6,)
                    if options.initial.shape != expected_shape:
                        raise ValueError('--initial value had the wrong number of components (got {0} but expected {1}).'.format(options.initial.shape, expected_shape))
                    options.qp_0 = options.initial.reshape(2,3)
                except ValueError as e:
                    print('error parsing --initial value: {0}'.format(str(e)))
                    self.op.print_help()
                    return None,None
            else:
                assert False, 'this should never happen because of the check with num_initial_conditions_specified'

        return options,args

