import heisenberg.library.orbit_plot
import heisenberg.util
import numpy as np
import optparse

class OptionParser:
    def __init__ (self, *, module):
        help_prolog = heisenberg.util.wrap_and_indent(text=module.subprogram_description, indent_count=1)
        self.__op = optparse.OptionParser(usage='%prog [options]\n\n{0}'.format(help_prolog))
        self.__op.prog = module.__package__
        self.__op.add_option(
            '--dt',
            dest='dt',
            default='0.001',
            type='float',
            help='Specifies the timestep for the curve integration.'
        )
        self.__op.add_option(
            '--max-time',
            dest='max_time',
            default='50.0',
            type='float',
            help='Specifies the max time to integrate the curve to.'
        )
        self.__op.add_option(
            '--seed',
            dest='seed',
            default=666,
            type='int',
            help='Specifies the seed to use for pseudorandom number generation.  Using the same seed should produce the same sequence of random numbers, and therefore provide reproducible program execution.'
        )
        self.__op.add_option(
            '--quantities-to-plot',
            dest='quantities_to_plot',
            type='str',
            default=heisenberg.library.orbit_plot.default_quantities_to_plot,
            help='Specifies which quantities to include in the plot.  Should be a semicolon-separated string, without spaces, with tokens selected from the following options: {0}.  Default is {1}'.format(';'.join(heisenberg.library.orbit_plot.valid_quantity_to_plot_v), heisenberg.library.orbit_plot.default_quantity_to_plot_v)
        )
        self.__op.add_option(
            '--disable-plot-decoration',
            dest='disable_plot_decoration',
            action='store_true',
            default=False,
            help='Disables plotting certain non-essential labels and decoration.  Default behavior is to plot those things.'
        )

        supported_plot_type_d = heisenberg.util.get_supported_plot_type_d()
        ext_v = sorted(list(supported_plot_type_d.keys()))

        self.__op.epilog = 'Available plot-types are:\n\n{0}'.format('\n'.join('    {0:4} : {1}'.format(ext, supported_plot_type_d[ext]) for ext in ext_v))

        help_string = 'Specifies the file type to use for plotting.  Filetypes depend on the particular backend in use by matplotlib.pyplot.  Available plot types are: {0}'.format(', '.join(ext_v))
        default_plot_type = 'png'
        assert default_plot_type in ext_v, '"png" not supported by the matplotlib.pyplot backend'
        self.__op.add_option(
            '--plot-type',
            dest='plot_type',
            choices=ext_v,
            default=default_plot_type,
            help=help_string
        )

    def parse_argv_and_validate (self):
        options,args = self.__op.parse_args()

        if options.dt is None:
            print('required option --dt was not specified.')
            self.__op.print_help()
            return None,None

        if options.max_time is None:
            print('required option --max-time was not specified.')
            self.__op.print_help()
            return None,None

        assert options.quantities_to_plot is not None
        print('options.quantities_to_plot = ', options.quantities_to_plot)
        quantity_to_plot_v = options.quantities_to_plot.split(';')
        if not frozenset(quantity_to_plot_v).issubset(frozenset(heisenberg.library.orbit_plot.valid_quantity_to_plot_v)):
            print('specified invalid elements in --quantities-to-plot: {0}'.format(','.join(frozenset(quantity_to_plot_v).difference(frozenset(heisenberg.library.orbit_plot.valid_quantity_to_plot_v)))))
            self.__op.print_help()
            return None,None
        options.quantity_to_plot_v = quantity_to_plot_v

        return options,args

    def add_option (self, *args, **kwargs):
        return self.__op.add_option(*args, **kwargs)

    def print_help (self):
        self.__op.print_help()
