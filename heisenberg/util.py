import heisenberg.library.util
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap

def construct_base_filename (*, obj, t_delta, t_max, initial_condition, t_min, k=None):
    base_filename = 'obj:{0:.4e}.dt:{1:.3e}.t_max:{2:.3e}.ic:{3}.t_min:{4:.4e}'.format(obj, t_delta, t_max, heisenberg.library.util.ndarray_as_single_line_string(initial_condition), t_min)
    if k is not None:
        base_filename = 'k:{0}.{1}'.format(k, base_filename)
    return base_filename

def random_embedding2_point (rng):
    C = 0.4
    #epsilon = 1.0e-5
    epsilon = 0.0
    # Perturb the bounds for p_x by epsilon away from the actual bound.
    return np.array([
        rng.uniform(-np.sqrt(1/(4*np.pi))+epsilon, np.sqrt(1/(4*np.pi))-epsilon),
        rng.uniform(0.0, C)
    ])

def pop_brackets_off_of (string):
    if len(string) < 2:
        raise ValueError('string (which is "{0}") must be at least 2 chars long'.format(string))
    elif string[0] != '[' or string[-1] != ']':
        raise ValueError('string (which is "{0}") must begin with [ and end with ]'.format(string))
    return string[1:-1]

def csv_as_ndarray (string, dtype):
    return np.array([dtype(token) for token in string.split(',')])

def wrap_and_indent (*, text, indent_count):
    return textwrap.indent(textwrap.fill(text, width=80), prefix=' '*(indent_count*4))

def ensure_dir_exists (path):
    os.makedirs(path, exist_ok=True)
    assert os.path.exists(path)

def find_next_output_dir (path_prefix):
    n = 0
    while True:
        output_dir = '{0}.{1:03}'.format(path_prefix, n)
        if not os.path.exists(output_dir):
            break
        elif n > 1000: # Cap it at some high limit so we don't loop too long
            raise Exception('couldn\'t find an output_dir in a reasonable amount of time; delete some of "output.*"')
        n += 1
    return output_dir

def get_supported_plot_type_d ():
    # This dict has the form { 'name of filetype':['ext', 'ext2'], 'name of another filetype':['ext3'] }
    supported_d = plt.gcf().canvas.get_supported_filetypes_grouped()
    # Change it so the keys are the file extensions, and the values are the names of the filetypes.
    retval_d = {}
    for name,ext_v in supported_d.items():
        for ext in ext_v:
            retval_d[ext] = name
    return retval_d
