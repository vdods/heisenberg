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
-   Replace the `options` param in the various subprograms' functions with specific params (maybe?)
-   Make a setup.py and define "entry points" for scripts (mainly `heisenberg` script):
    https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
-   Make 3d plot option.
-   Optionally use other sheet of H=0 in solution for p_z.  Generally allow different coordinate charts.
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
-   Design for external program interface:

    Have the following modules each be separate [sub-]programs:

        heisenberg                  (heisenberg/__main__.py)
        heisenberg.plot             (heisenberg/plot/__main__.py)
        heisenberg.search           (heisenberg/search/__main__.py)
        heisenberg.sample           (heisenberg/sample/__main__.py)
        heisenberg.plot_samples     (heisenberg/plot_samples/__main__.py)

    Where heisenberg just prints usage information for the sub-programs.  There is a set of common options,
    and then there are sub-program specific options, which should be defined in and parsed by each
    sub-program.

    The sub-programs can be invoked as

        python3 -m heisenberg
        python3 -m heisenberg.plot
        python3 -m heisenberg.search
        python3 -m heisenberg.sample
        python3 -m heisenberg.plot_samples

    Or once setup.py is defined with its entry points, the following scripts will be created that can be
    executed:

        heisenberg
        heisenberg.plot
        heisenberg.search
        heisenberg.sample
        heisenberg.plot_samples
-   Make subprograms for reproducing specific results.
"""

# TODO: Make a description string in each executable module and have this program just print them.
print("""
Heisenberg:

    Numerical experiments for finding closed (i.e. periodic) orbits in the Kepler problem on the Heisenberg group.

Authors:

    Victor Dods and Corey Shanbrom

Sub-programs:

    heisenberg              This program, which is simply a directory for the various sub-programs.

    heisenberg.plot         Plot an integral curve of the system using given initial condition.

    heisenberg.plot_samples Provides visualization of the data generated by the heisenberg.sample
                            subprogram.  In particular, this gives a colormapped scatterplot of the
                            objective function on the fully reduced, 2-parameter initial condition space.

    heisenberg.search       Search a specified parameter space for initial conditions for curves.  Once
                            a curve is found that comes close enough to closing back up on itself, an
                            optimization method is used to attempt to alter the initial conditions so
                            that the curve closes back up on itself.  The result is integral curves
                            which approximate closed solutions to within numerical accuracy.

    heisenberg.sample       Samples a specified parameter space of initial conditions, computing the
                            corresponding integral curves, and computing and storing relevant data
                            about the each curve in a file.  This is intended to sample various functions
                            of the initial condition space, and can be used in later processing;
                            see the heisenberg.plot_samples subprogram.
""")
