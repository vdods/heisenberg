import cmath
import itertools
import math
import numpy as np
import tensor

def realify_complex_matrix (M):
    retval = np.ndarray((2*M.shape[0],2*M.shape[1]), dtype=float)
    for r,row in enumerate(M):
        R = 2*r
        for c,component in enumerate(row):
            C = 2*c
            retval[R,C]     =  component.real
            retval[R,C+1]   = -component.imag
            retval[R+1,C+1] =  component.real
            retval[R+1,C]   =  component.imag
    return retval

def realify_complex_vector (Z):
    retval = np.ndarray((2*X.shape[0],), dtype=float)
    for i,z in enumerate(Z):
        I = 2*i
        retval[I] = z.real
        retval[I+1] = z.imag
    return retval

class Transform:
    def __init__ (self, modes, sample_times, period):
        assert len(modes) > 0
        assert len(sample_times) >= len(modes), 'Must have at least as many samples as modes.'
        assert len(sample_times) > 0 # This is just a sanity check, but logically follows from above assertions.
        assert all(sample_times[i+1] > sample_times[i] for i in range(len(sample_times)-1)), 'sample_times must be a strictly increasing sequence.'
        assert period > sample_times[-1], 'period must be greater than last element of sample_times.'

        self.modes = modes
        self.sample_times = sample_times
        self.period = period

        omega = 2.0*math.pi/self.period
        
        i_omega = 1.0j*omega
        dts = [sample_times[i+1] - sample_times[i] for i in range(len(sample_times)-1)] + [period - sample_times[-1]]
        assert all(dt > 0 for dt in dts)
        assert len(dts) == len(sample_times)

        def squared_L2_norm_complex (T):
            return sum(z.real**2 + z.imag**2 for z in T.flat) / T.size

        def squared_L2_norm_real (T):
            return sum(t**2 for t in T.flat) / T.size

        coeffs_from_samples_matrix = np.array([[cmath.exp(-i_omega*mode*sample_time)*dt/period for sample_time,dt in itertools.izip(sample_times,dts)] for mode in modes])
        samples_from_coeffs_matrix = np.array([[cmath.exp(i_omega*mode*sample_time) for mode in modes] for sample_time in sample_times])
        assert squared_L2_norm_complex(np.einsum('ij,jk', coeffs_from_samples_matrix, samples_from_coeffs_matrix) - np.eye(len(modes))) < 1.0e-10
        time_derivative_matrix = np.diag([i_omega*mode for mode in modes])
        time_integral_periodic_part_matrix = np.diag([1.0/(i_omega*mode) if mode != 0 else 0.0 for mode in modes])
        assert squared_L2_norm_complex(np.einsum('ij,jk', time_derivative_matrix, time_integral_periodic_part_matrix) - np.diag([1.0 if mode != 0 else 0.0 for mode in modes])) < 1.0e-10

        self.real_coefficient_count = 2*len(modes)
        self.sample_count = len(sample_times)

        self.coeffs_from_samples_matrix = realify_complex_matrix(coeffs_from_samples_matrix)
        self.samples_from_coeffs_matrix = realify_complex_matrix(samples_from_coeffs_matrix)
        assert squared_L2_norm_real(np.einsum('ij,jk', self.coeffs_from_samples_matrix, self.samples_from_coeffs_matrix) - np.eye(self.real_coefficient_count)) < 1.0e-10
        self.time_derivative_matrix = realify_complex_matrix(time_derivative_matrix)
        self.time_integral_periodic_part_matrix = realify_complex_matrix(time_integral_periodic_part_matrix)

if __name__ == '__main__':
    import symbolic
    import time

    # This and einsum_for_two is from http://stackoverflow.com/questions/15606937/how-do-i-get-numpy-einsum-to-play-well-with-sympy
    def einsum(string, *args):
        index_groups = map(list, string.split(','))
        assert len(index_groups) == len(args)
        tensor_indices_tuples = zip(index_groups, args)
        return reduce(einsum_for_two, tensor_indices_tuples)[1]
    def einsum_for_two(tensor_indices1, tensor_indices2):
        string1, tensor1 = tensor_indices1
        string2, tensor2 = tensor_indices2
        sum_over_indices = set(string1).intersection(set(string2))
        new_string = string1 + string2
        axes = ([], [])
        for i in sum_over_indices:
            new_string.remove(i)
            new_string.remove(i)
            axes[0].append(string1.index(i))
            axes[1].append(string2.index(i))
        return new_string, np.tensordot(tensor1, tensor2, axes)

    sym_class = 1
    fold = 5
    k = 4
    modes = range(sym_class-fold*k,sym_class+fold*k+1,fold)
    sample_count = 1000
    period = 10.0
    sample_times = np.linspace(0.0, period, sample_count+1)[:-1]
    start = time.time()
    ft = Transform(modes, sample_times, period)
    end = time.time()
    print 'duration of ft generation = {0} s'.format(end - start)

    R = np.random.randn(ft.real_coefficient_count)
    # R = symbolic.tensor('R', (ft.real_coefficient_count,))
    start = time.time()
    # samples = tensor.contract('ij,j', ft.samples_from_coeffs_matrix, R, dtype=float)
    samples = np.einsum('ij,j', ft.samples_from_coeffs_matrix, R)
    # samples = einsum('ij,j', ft.samples_from_coeffs_matrix, R)
    end = time.time()
    print 'duration of curve reconstruction = {0} s'.format(end - start)

    import matplotlib.pyplot as plt
    X = samples[0::2]
    Y = samples[1::2]
    plt.plot(X,Y)
    plt.savefig('giraffe.png')

