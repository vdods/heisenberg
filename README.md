# Kepler-Heisenberg Problem Computational Tool Suite

Code for numerical computation in research on orbital dynamics in the Kepler-Heisenberg problem
(solving for sun/planet dynamics in the Heisenberg group) -- a collaboration with Corey Shanbrom.

The publication "Numerical Methods and Closed Orbits in the Kepler-Heisenberg Problem" in which these results appear can
be found in [the Journal of Experimental Mathematics](http://www.tandfonline.com/doi/full/10.1080/10586458.2017.1416709),
and the preprint on [the arXiv](https://arxiv.org/abs/1707.05937).

## Contributors

- Victor Dods    : programming, some math
- Corey Shanbrom : math, some programming

## Results Reproduction

The principal aim in publishing this code repository in tandem with the peer-reviewed publication is
to provide the ability to freely, easily, and exactly reproduce the the results therein.  Ideally,
this will contribute to a higher standard of what is expected from a publication having a computer-based
element -- it should be as easy as is reasonably possible for an interested reader to reproduce the
results.

Full details on how to reproduce the results found in "Numerical Methods and Closed Orbits in the Kepler-Heisenberg Problem"
can be found [here](NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/README.md).

## License and Attribution

This project is held under Copyright (2014-2017) by Victor Dods and is released as free, open-source software
under the [MIT License](LICENSE.md), without warranty.

If you use any code from this project, I'd love to hear about what you're applying it to!  Send me an email about it at
`victor <dot> dods <at-sign> gmail <dot> com`

It is requested, though not required, that any use of the code from this project in a derivative work be cited/acknowledged
in the relevant documentation for the derivative work.  The citation should include:
-   Author: `Victor Dods`
-   The name of this project: `Kepler-Heisenberg Problem Computational Tool Suite`
-   The public link to this github repository: `https://github.com/vdods/heisenberg`

## Contents

-   [heisenberg](https://github.com/vdods/heisenberg/tree/master/heisenberg) : Top-level program
    (and module) to explain and redirect to all the specific subprograms of this project.
-   [heisenberg.library](https://github.com/vdods/heisenberg/tree/master/heisenberg/library) :
    A module that contains mostly math-oriented code that is used in multiple subprograms.
-   [heisenberg.plot](https://github.com/vdods/heisenberg/tree/master/heisenberg/plot) :
    Plots an integral curve of the system using given initial condition, optionally running an
    optimization method to find a nearby curve that closes back up on itself.
-   [heisenberg.plot_samples](https://github.com/vdods/heisenberg/tree/master/heisenberg/plot_samples) :
    Provides visualization of the data generated by the heisenberg.sample subprogram.  In particular,
    this gives a colormapped scatterplot of the objective function on the fully reduced, 2-parameter
    initial condition space.
-   [heisenberg.sample](https://github.com/vdods/heisenberg/tree/master/heisenberg/sample) :
    Samples a specified parameter space of initial conditions, computing the corresponding integral
    curves, and computing and storing relevant data about the each curve in a file.  This is intended
    to sample various functions of the initial condition space, and can be used in later processing;
    see the heisenberg.plot_samples subprogram.
-   [heisenberg.search](https://github.com/vdods/heisenberg/tree/master/heisenberg/search) :
    Search a specified parameter space for initial conditions for curves.  Once a curve is found
    that comes close enough to closing back up on itself, an optimization method is used to attempt
    to alter the initial conditions so that the curve closes back up on itself.  The result is
    integral curves which approximate closed solutions to within numerical accuracy.

Additional content that is no longer an active part of the project:

-   [attic](https://github.com/vdods/heisenberg/tree/master/attic) : Directory containing deprecated code.

## Invoking Subprograms

The following command, when executed from this project's root directory, will print a directory of
the available subprograms.

    python3 -m heisenberg

Examples of invoking specific subprograms:

    python3 -m heisenberg.plot --dt=0.003 --max-time=25.03042826445711 --initial-preimage=[0.2706994702908095] --embedding-dimension=1 --embedding-solution-sheet-index=0 --output-dir=generated-data --disable-plot-decoration --cut-off-initial-curve-tail --quantities-to-plot=x,y --plot-type=pdf

    python3 -m heisenberg.search --dt=1.0e-2 --max-time=200 --seed=123456789 --embedding-dimension=2 --embedding-solution-sheet-index=1 --plot-type=pdf --output-dir=generated-data --exit-after-number-of-successes=1 --quantities-to-plot="x,y;t,z;sqd;objective" --use-terse-plot-titles --plot-size=3

Please see [this](NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/README.md) and the scripts that it
refers to for more specific examples.

## Installing the `vorpy` Dependency

The `vorpy` Python module is needed by this tool suite.  To install the latest version available on github, simply
run the following command.

    pip install --upgrade git+https://github.com/vdods/vorpy.git

Version `0.4.1` was the particular version used during the development and research for this project.

## To-do List

A software project is never truly finished.  See the list of to-dos [here](TODO.md).
