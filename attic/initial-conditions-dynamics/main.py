# Add our library code directory to the import search path.  This allows things like
# `import fourier` to work.  It relies on the relative filesystem position of this
# file to the `library` directory.
import sys
sys.path.append('../library')

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
    reconstructed_Ws = Ft.sampled_sum_of(coefficients)

    import itertools
    # Really the norm should be taken via the norm defined by a Riemann sum that involves Ft.dts_for_average.
    print("|reconstructed_Ws - Ws|^2 = {0}".format(linalg_util.ComplexVectorNormSquared((reconstructed_Ws - Ws)/len(Ft.sample_times))))
    diffs = [linalg_util.ComplexNormSquared(reconstructed_W - W) for (reconstructed_W,W) in itertools.izip(reconstructed_Ws,Ws)]
    max_diff = max(diffs)
    print("max_diff = {0}".format(max_diff))

def main ():
    import heisenberg_dynamics
    (Xs,Ts,period) = heisenberg_dynamics.compute_coreys_flow_curve()

    # Compute the Hamiltonian at each curve sample point to verify that energy is
    # identically zero.  Do this by computing the max of the norm of the energy.
    Hs = [heisenberg_dynamics.hamiltonian(X) for X in Xs]
    max_abs_H = max(abs(H) for H in Hs)
    print('max_abs_H = {0}'.format(max_abs_H))

    # Plot the curve to the file
    import plotty
    plotty.plotty_2d_points(Xs, save_to_file='dynamics_rk4_x.png')

    pickle_curve_positions(Xs, Ts, period)

    compute_Fourier_coefficients_of(Xs, Ts)

if __name__ == "__main__":
    main()

