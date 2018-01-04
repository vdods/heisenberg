# TODO List

These items have been accumulated over the course of the development of this project, and are not likely
to ever be completely finished.  One can consider software development to be an asymptotic process.

-   Use ast.literal_eval (as in heisenberg.sample) to parse initial conditions options.
-   Idea for cleaner and more organized option specification and validation:
    -   Make a class which adds each individual option and then can post-process and validate it after the
        options have been parsed by optparse.  This way, each subprogram can just list which ones it
        wants, and all the post-processing and validation is taken care of automatically (and in a
        separate place)
-   Move all options validation into `__main__.py` in each subprogram, since that's where it's defined
    and mostly handled.
-   Make plotting interactive (pyplot vs pyqtgraph)
-   Figure out how to use pyqtgraph for static plot generation with no windows or interaction.
-   Figure out why OptionParser epilog strips out newlines.
-   Make it so that heisenberg subprogram modules can be imported lightly -- so they don't import anything.
-   Replace the `options` param in the various subprograms' functions with specific params (maybe?)
-   Make a setup.py and define "entry points" for scripts (mainly `heisenberg` script):
    https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
-   Make 3d plot option.
-   optparse is apparently deprecated -- switch to argparse https://docs.python.org/3/howto/argparse.html
-   Create a quadratic_min_time_parameterized which takes more than 3 points and does a least-squares fit.
-   Change the search mechanism for increasing time such that no redundant integration is done, but
    simply continue the integration at the time it left off.
-   Examine following hypothesis: Z curve for quasi-periodic orbit is a sine wave (or whatever periodic
    function) of decreasing amplitude and increasing frequency (or vice versa, depending on if the orbit
    is spiraling in or out), perhaps this can be expressed simply as being a rotated and dilated (and
    time-reparameterized) version of itself.  To put more concisely, examine the hypothesis that
    quasi-periodic curves are self-similar when segmented using some "sub/quasi-period".
-   If abs(H) escapes a certain threshold while integrating, either abort or decrease the timestep.
    This may be a more appropriate TODO for `vorpy`, since that's what implements the symplectic
    integrator.
-   Decide dt based on the abs(H) bound -- if it's above say 1.0e-4, then reduce dt.  Again, this may
    be more appropriate for `vorpy`.
-   k-fold symmetry -- make the closest-approach-map measure from a 2pi/k-rotated phase space point
    in order to more efficiently find k-fold symmetric curves.
-   Idea for defining a more-robust objective function: Compute t_min as before, but then define the
    objective function as the L_2 norm squared of the difference between overlapping segments.
    In more detail, the candidate curve extends past t_min, say until t_min+T for some positive T.
    Then define the objective function to be

        L_2(curve[0:T] - curve[t_min:t_min+T])^2

    so that curves that are nearly coincident for a positive time duration have a lower objective
    function than curves that merely have a single point that comes close to the initial condition.
-   Minimal, quick design for "nice plots"
    -   Nice to have: SVG output of (x,y,z) curve in 3D.
-   Add following info to output of summary files:
    -   Duration of execution
    -   What processor it was run on, how many cores, threads, how much memory was used (max), etc.
    -   Some indication of if it had to share processing time with other significant processes
    -   Amount of disk space used by files written
-   The specifics of quasi-periodic orbits in the K-H problem make it such that perhaps a better way of defining
    the objective function would be to find where the orbit intersects the z=0 plane in the same direction as the
    initial condition, and compute the squared distances in phase space of those points with the initial condition.
    The lowest of these values would be the objective function value.  The point of this is that it would be using
    the analogous point on a different lobe of the curve to compare.  This may also make computing a sub-sample
    distance easier and more accurate.
