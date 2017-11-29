# TODO List

These items have been accumulated over the course of the development of this project, and are not likely
to ever be completely finished.  One can consider software development to be an asymptotic process.

-   Create "turn-key" examples which replicate research by running a single command.
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
-   Make subprograms for reproducing specific results -- make a subprogram called `reproduce` or something
    and have it have options to reproduce each figure, or at least print exact directions on how to reproduce them,
    including which git commit hash was used to generate the figures in the published paper, and the commands
    that generated them.
-   Minimal, quick design for "nice plots"
    -   Want SVG output of just (x,y) curve, with no axes, but with plotting the sun (origin) and initial point.
    -   Nice to have: SVG output of (x,y,z) curve in 3D.
-   Bug: In heisenberg.search, bug encountered in abortive condition:

        curve did not nearly close up -- retrying with higher t_max: 192.216796875
        t_max (192.216796875) was raised too many times, exceeding --max-time value of 192.2, before nearly closing up -- aborting
        strength-sorted modes (only the first few):
            0 : 1.719029e+02 (freq = 0)
            1 : 1.546591e+02 (freq = 1)
            2 : 1.987003e+01 (freq = 2)
            3 : 1.083982e+01 (freq = 3)
            4 : 7.366250e+00 (freq = 4)
        encountered exception of type <class 'ValueError'> during try_random_initial_condition; skipping.  exception was: Invalid number of FFT data points (0) specified.
        stack:
        File "/home/vdods/files/github/vdods/heisenberg/heisenberg/search/__init__.py", line 157, in search
            try_random_initial_condition()
        File "/home/vdods/files/github/vdods/heisenberg/heisenberg/search/__init__.py", line 74, in try_random_initial_condition
            symmetry_class_estimate=smo_0.symmetry_class_estimate(),
        File "/home/vdods/files/github/vdods/heisenberg/heisenberg/library/shooting_method_objective.py", line 225, in symmetry_class_estimate
            abs_fft_xy_rfc  = np.abs(self.fft_xy_resampled_flow_curve(sample_count=scipy.fftpack.next_fast_len(order_estimate*mode_wraps)))
        File "/home/vdods/files/github/vdods/heisenberg/heisenberg/library/shooting_method_objective.py", line 154, in fft_xy_resampled_flow_curve
            fft_xy_rfc = scipy.fftpack.fft(xy_rfc)
        File "/usr/local/lib/python3.5/dist-packages/scipy/fftpack/basic.py", line 282, in fft
            "(%d) specified." % n)

-   Add following info to output of summary files:
    -   Duration of execution
    -   What processor it was run on, how many cores, threads, how much memory was used (max), etc.
    -   Some indication of if it had to share processing time with other significant processes
    -   Amount of disk space used by files written
-   The 1:1 orbit in the paper that looks like an infinity sign is classified by the class:order estimators as 0:1.
    While 0:1 and 1:1 are technically both correct, a single convention should be chosen.  Perhaps the initial
    condition for the 1:1 should be used to decide (does it show up nicely in the table of j/k <--> initial p_theta?)
-   Take out --k-fold-initial
