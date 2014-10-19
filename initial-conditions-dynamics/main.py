import sys

# Add our library code directory to the import search path.  This allows things like
# `import Fourier` to work.  It relies on the relative filesystem position of this
# file to the `library` directory.
sys.path.append('../library')

def compute_coreys_flow_curve ():
    period = 273.5
    sample_count = 2735
    import heisenberg_dynamics
    X_0 = heisenberg_dynamics.coreys_initial_condition()
    print "Computing flow curve for time duration {0}, with {1} samples, from initial condition {2}.".format(period, sample_count, X_0)
    import vector_field
    (Xs,Ts) = vector_field.compute_flow_curve(heisenberg_dynamics.hamiltonian_vector_field, X_0, 0.0, period, sample_count)
    return (Xs,Ts,period)

def pickle_curve_positions (Xs, Ts, period):
    import numpy
    samples = [numpy.array(X[0:3], dtype=float) for X in Xs]
    sample_times = Ts
    data = [sample_times, samples, period]

    import pickle
    pickle.dump(data, open('sample_times,samples,period.pickle', 'wb'))

def compute_Fourier_coefficients_of (Xs, Ts):
    L = 60
    import fourier
    Ft = fourier.Transform(range(-L,L+1), Ts)

    import linalg_util
    # Extract just the (x,y) components of the curve sample points' position
    # as complex values (call these values Ws, each element being a W) as the
    # z component of the curve is a function of the (x,y) components.
    Ws = linalg_util.ComplexVector([complex(X[0],X[1]) for X in Xs[0:-1]])
    coefficients = Ft.coefficients_of(Ws)
    # print 'coefficients = {0}'.format(coefficients)
    reconstructed_Ws = fourier.sampled_sum_of(coefficients)

    import itertools
    # Really the norm should be taken via the norm defined by a Riemann sum that involves Ft.dts_for_average.
    print "|reconstructed_Ws - Ws|^2 = {0}".format(linalg_util.ComplexVectorNormSquared((reconstructed_Ws - Ws)/len(Ft.sample_times)))
    diffs = [linalg_util.ComplexNormSquared(reconstructed_W - W) for (reconstructed_W,W) in itertools.izip(reconstructed_Ws,Ws)]
    max_diff = max(diffs)
    print "max_diff = {0}".format(max_diff)

def test_Fourier ():
    import sys

    L_range = range(3,40)
    sample_count_range = range(10,100+1)

    sys.stdout.write('horizontal axis is mode count, vertical axis is sample count.\n')
    sys.stdout.write('{:>20} '.format(''))
    for L in L_range:
        mode_count = L
        sys.stdout.write('{:>20}'.format(mode_count))
    sys.stdout.write('\n\n')
    for sample_count in sample_count_range:
        sys.stdout.write('{:>20}:'.format(sample_count))
        for L in L_range:
            import fourier
            sys.stdout.write('{:>20}'.format(fourier.Transform.test(L, sample_count)))
        sys.stdout.write('\n')

def main ():
    # test_Fourier()

    (Xs,Ts,period) = compute_coreys_flow_curve()

    # Compute the Hamiltonian at each curve sample point to verify that energy is
    # identically zero.  Do this by computing the max of the norm of the energy.
    import heisenberg_dynamics
    Hs = [heisenberg_dynamics.hamiltonian(X) for X in Xs]
    max_abs_H = max(abs(H) for H in Hs)
    print 'max_abs_H = {0}'.format(max_abs_H)

    # Plot the curve to the file
    import plotty
    plotty.plotty_2d_points(Xs, save_to_file='dynamics_rk4.png')

    pickle_curve_positions(Xs, Ts, period)

if __name__ == "__main__":
    main()

