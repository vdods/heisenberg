import library.util
import numpy as np

def construct_base_filename (*, obj, t_delta, t_max, initial_condition, t_min):
    return 'obj:{0:.4e}.dt:{1:.3e}.t_max:{2:.3e}.ic:{3}.t_min:{4:.4e}'.format(obj, t_delta, t_max, library.util.ndarray_as_single_line_string(initial_condition), t_min)

def random_embedding2_point (rng):
    C = 2.0
    #epsilon = 1.0e-5
    epsilon = 0.0
    # Perturb the bounds for p_x by epsilon away from the actual bound.
    return np.array([
        rng.uniform(-np.sqrt(4/np.pi)+epsilon, np.sqrt(4/np.pi)-epsilon),
        rng.uniform(0.0, C)
    ])

