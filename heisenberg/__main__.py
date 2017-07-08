"""
Notes

Define "return map" R : T^* Q -> T^* Q (really R^3xR^3 -> R^3xR^3, because it's coordinate dependent):
R(q,p) is defined as the closest point (in the coordinate chart R^3xR^3 for T^* Q) to (q,p) in the
sequence of points in the solution to the orbital curve for initial condition (q,p).

Define f : T^* Q -> R, (q,p) |-> 1/2 * |(q,p) - R(q,p)|^2

Use an optimization method (e.g. a Monte Carlo method or gradient descent) to find critical points of f.

The gradient of f depends on the gradient of R.  This can be computed numerically using a least-squares
approximation of the first-order Taylor polynomial of R.

Select initial conditions for the gradient descent to be on the H(q,p) = 0 submanifold, probably
by picking 5 coordinates at random and solving for the 6th.

Symmetry condition:  Define symmetry via map Omega : T^* Q -> T^* Q (e.g. rotation through 2*pi/3).
Define R_Omega to give point closest to Omega(q,p).  Then f_Omega is defined as

    f_Omega(q,p) := 1/2 * |Omega(q,p) - R_Omega(q,p)|^2,

and the gradient of f_Omega depends on the gradient of Omega and R_Omega.

TODO
-   optparse is apparently deprecated -- switch to argparse https://docs.python.org/3/howto/argparse.html
-   Take all dynamics_context-specific code out of OptionParser
-   Create a quadratic_min_time_parameterized which takes more than 3 points and does a least-squares fit.
-   Ensure that the objective function as computed in ShootingMethodObjective is nonnegative.
-   Change the search mechanism for increasing time such that no redundant integration is done, but
    simply the integrator continues at the time it left off.
-   In addition to plot and pickle files, output a short, human-readable summary of the pickle file,
    most importantly with the exact initial conditions, dt, t_min, and the values for alpha and beta.
-   Examine following hypothesis: Z curve for closed orbit is a sine wave.
-   Examine following hypothesis: Z curve for quasi-periodic orbit is a sine wave (or whatever periodic
    function) of decreasing amplitude and increasing frequency (or vice versa, depending on if the orbit
    is spiraling in or out), perhaps this can be expressed simply as being a rotated and dilated (and
    time-reparameterized) version of itself.
-   If abs(H) escapes a certain threshold while integrating, either abort or decrease the timestep.
-   k-fold symmetry -- make the closest-approach-map measure from a 2pi/k-rotated phase space point
    in order to more efficiently find k-fold symmetric curves.
-   Decide dt based on the abs(H) bound -- if it's above say 1.0e-4, then reduce dt.
-   When plotting closed curves, should probably only plot up to the t_min value, not the overlapping
    part.
-   Idea for defining a more-robust objective function: Compute t_min as before, but then define the
    objective function as the L_2 norm squared of the difference between overlapping segments.
    In more detail, the candidate curve extends past t_min, say until t_min+T for some positive T.
    Then define the objective function to be

        L_2(curve[0:T] - curve[t_min:t_min+T])^2

    so that curves that are nearly coincident for a positive time duration have a lower objective
    function than curves that merely have a single point that comes close to the initial condition.
-   Program to search for orbits in the 2D embedding space:
    -   Generate random points in the domain

            -sqrt(4/pi) <= p_x <= sqrt(4/pi)
                     -C <= p_y <= C

        for some arbitrary positive constant C, say 2.  Due to a discrete symmetry in the system
        (reflection), p_y can be taken to be nonnegative.  Thus the domain can be

            -sqrt(4/pi) <= p_x <= sqrt(4/pi)
                      0 <= p_y <= C

    -   For each of these, compute the embedding qp_0 into phase space and use that as the initial
        condition for the flow curve.  Use some fixed dt and max_time.
    -   For each flow curve, compute the following values:
        -   The objective function value (which could be NaN if the curve didn't go back toward itself)
        -   The t_min value (which for a closed curve would be its period)
        -   The upper bound on abs(H), or more detailed stats (mean and variance, or a histogram)
        -   The upper bound on abs(J - J_0), or more detailed stats (mean and variance, or a histogram)
        -   If it is [close to being] closed, then the order of its radial symmetry
        -   The deviation of its z(t) function from a sine wave (for closed, this sometimes but not
            always appears to be a sine wave, but this hasn't been checked rigorously).
    -   Visualize these two functions to determine if they are continuous and/or have other structure
        to them.  Perhaps there are regions which are continuous surrounding zeros the objective function.
"""

if __name__ == '__main__':
    import sys

    sys.path.append('..') # TEMP HACK so that the library module is accessible from here.

    import library.heisenberg_dynamics_context
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from . import option_parser
    from . import plot
    from . import sample
    from . import search

    # https://github.com/matplotlib/matplotlib/issues/5907 says this should fix "Exceeded cell block limit" problems
    matplotlib.rcParams['agg.path.chunksize'] = 10000

    dynamics_context = library.heisenberg_dynamics_context.HeisenbergDynamicsContext_Numeric()

    op = option_parser.OptionParser()
    options,args = op.parse_argv_and_validate(sys.argv, dynamics_context)
    if options is None:
        sys.exit(-1)

    print('options: {0}'.format(options))
    print('args   : {0}'.format(args))

    rng = np.random.RandomState(options.seed)

    if options.search:
        search.search(dynamics_context, options, rng=rng)
    elif options.sample:
        sample.sample(dynamics_context, options, rng=rng)
    elif options.plot:
        plot.plot(dynamics_context, options, rng=rng)
    else:
        print('Must specify one of --search, --sample, or --plot.')
        sys.exit(-1)
