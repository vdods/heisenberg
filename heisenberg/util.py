import heisenberg.library.util
import numpy as np
import textwrap

def construct_base_filename (*, obj, t_delta, t_max, initial_condition, t_min):
    return 'obj:{0:.4e}.dt:{1:.3e}.t_max:{2:.3e}.ic:{3}.t_min:{4:.4e}'.format(obj, t_delta, t_max, heisenberg.library.util.ndarray_as_single_line_string(initial_condition), t_min)

def random_embedding2_point (rng):
    C = 2.0
    #epsilon = 1.0e-5
    epsilon = 0.0
    # Perturb the bounds for p_x by epsilon away from the actual bound.
    return np.array([
        rng.uniform(-np.sqrt(4/np.pi)+epsilon, np.sqrt(4/np.pi)-epsilon),
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
