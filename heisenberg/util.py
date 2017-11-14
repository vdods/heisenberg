import heisenberg.library.util
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import textwrap

def construct_base_filename (*, symmetry_order_estimate, symmetry_class_estimate, obj, t_delta, t_max, initial_condition, sheet_index, t_min):
    base_filename = 'obj:{0:.4e}.dt:{1:.3e}.t_max:{2:.3e}.ic:{3}.sheet_index:{4}.t_min:{5:.4e}'.format(obj, t_delta, t_max, heisenberg.library.util.ndarray_as_single_line_string(initial_condition), sheet_index, t_min)
    # Prepend symmetry_class_estimate
    if symmetry_class_estimate is not None:
        base_filename = 'class:{0}.{1}'.format(symmetry_class_estimate, base_filename)
    # order goes first because it's a more significant classification.
    if symmetry_order_estimate is not None:
        base_filename = 'order:{0}.{1}'.format(symmetry_order_estimate, base_filename)
    return base_filename

__indent = '    '

def write_human_readable_value (f, name, value, indent_level=0):
    if type(value) == dict:
        write_human_readable_dict(f, name, value, indent_level)
    elif type(value) == list:
        write_human_readable_list(f, name, value, indent_level)
    elif type(value) == np.ndarray:
        write_human_readable_list(f, name, value.tolist(), indent_level, type_name_override='numpy.ndarray')
    else:
        f.write(__indent*indent_level)
        if name is not None:
            f.write(name+' : ')
        f.write(str(value)+'\n')

def write_human_readable_dict (f, dict_name, dict_value, indent_level=0):
    label = dict_name+' ' if dict_name is not None else ''
    f.write(__indent*indent_level)
    if dict_name is not None:
        f.write(dict_name+' ')
    f.write('(dict; {0} items):\n'.format(len(dict_value)))
    # Get the keys in sorted order
    key_v = sorted(list(dict_value.keys()))
    for key in key_v:
        value = dict_value[key]
        write_human_readable_value(f, repr(key), value, indent_level+1)

__max_print_list_element_count = 10

def write_human_readable_list (f, list_name, list_value, indent_level=0, type_name_override=None):
    f.write(__indent*indent_level)
    if list_name is not None:
        f.write(list_name+' ')
    type_name = type_name_override if type_name_override is not None else 'list'
    f.write('({0}; {1} items):\n'.format(type_name, len(list_value)))
    if len(list_value) > __max_print_list_element_count:
        assert __max_print_list_element_count//2 + (__max_print_list_element_count+1)//2 == __max_print_list_element_count
        for value in list_value[:__max_print_list_element_count//2]:
            write_human_readable_value(f, None, value, indent_level+1)
        write_human_readable_value(f, None, '... (excessive items omitted) ...', indent_level+1)
        for value in list_value[-((__max_print_list_element_count+1)//2):]:
            write_human_readable_value(f, None, value, indent_level+1)
    else:
        for value in list_value:
            write_human_readable_value(f, None, value, indent_level+1)

def write_human_readable_summary (*, data, filename):
    """
    I would use JSON here, but the json module doesn't automatically serialize numpy.ndarray,
    which is pretty stupid.
    """
    with open(filename, 'wt') as f:
        f.write('human-readable summary for\n')
        f.write(filename)
        f.write('\n\n')
        write_human_readable_value(f, 'pickle contents', data)

def get_git_commit ():
    try:
        working_copy_has_changed    = subprocess.getstatusoutput('git diff --quiet')[0] != 0
        git_describe                = subprocess.check_output(['git', 'describe', '--abbrev=40', '--always']).decode('utf-8').strip()
        if working_copy_has_changed:
            git_describe += '-with-changed-working-copy'
        return git_describe
    except Exception as e:
        return '<could not retrieve current git commit>'

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
