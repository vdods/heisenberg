# Add our library code directory to the import search path.  This allows things like
# `import fourier` to work.  It relies on the relative filesystem position of this
# file to the `library` directory.
import sys
sys.path.append('library')

import pickle
(sample_times,samples,period) = pickle.load(open('initial-conditions-dynamics/sample_times,samples,period.pickle', 'rb'))

import linalg_util
Zs = linalg_util.FloatVector([sample[2] for sample in samples[0:-1]])

# import plotty
# plotty.plotty_2d_points([(t,z) for (t,z) in zip(sample_times,Zs)], save_to_file='z.png')

# import fourier
# Z_modes = range(-30,30+1)
# Ft = fourier.Transform(Z_modes, sample_times)
# Z_coefficients = Ft.coefficients_of(Zs)
# reconstructed_Zs = Ft.fourier_sum_of(Z_coefficients)

# import itertools
# for (z,r) in itertools.izip()

Z_modes = [1,3,5,7,9,11]

import numpy
Ts = numpy.linspace(0.0, 1.0, num=401)

import fourier
Ft = fourier.Transform(Z_modes, Ts)

import linalg_util
Z_coefficients = linalg_util.ComplexVector([0.2, 0.5, 0.1, 0.4, -0.3, -0.1])
Zs = Ft.sampled_sum_of(Z_coefficients)

import itertools
import plotty
plotty.plotty_2d_points([(t,z.real) for (t,z) in itertools.izip(Ts,Zs)], save_to_file='z.png');

for t in Ts:
	if t < 0.5:
		one_minus_t = 1.0 - t
		z_t_plus_P_over_2 = Ft.evaluate_sum_at_arbitrary_time(Z_coefficients, t+Ft.period/2.0)
		negative_z_t = -Ft.evaluate_sum_at_arbitrary_time(Z_coefficients, t)
		diff_norm_squared = linalg_util.ComplexNormSquared(z_t_plus_P_over_2 - negative_z_t)
		print 't = {0}, diff_norm_squared = {1}'.format(t, diff_norm_squared)



