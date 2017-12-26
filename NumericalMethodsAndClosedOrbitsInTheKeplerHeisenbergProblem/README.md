# Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem

This computational tool suite was created to conduct numerical experiments for research which resulted in the acceptance of
the paper "Numerical Methods and Closed Orbits in the Kepler–Heisenberg Problem" ([preprint](https://arxiv.org/abs/1707.05937))
in the journal [Experimental Mathematics](http://www.tandfonline.com/loi/uexm20).

Great effort went into the development of this tool suite, and a corresponding amount of effort went into ensuring that
the experimental results obtained herein are entirely reproducible by anyone with internet access and access to modern
programming tools.  This codebase is open-source and may be used free of charge.  See the [LICENSE.md](../LICENSE.md) file
for details.

## Result Reproduction Instructions

First, it should be noted that the results for this paper were generated from the state of the git repository
tagged `NumericalMethods2017`, so before proceeding, make sure you've checked out that tag using the following
command from the project root (the directory which contains the file `.gitignore` and `LICENSE.md`).

    git checkout NumericalMethods2017

Alternately, to retrieve the correct version without using the `git` commandline tool at all, download and
extract the tagged archive using the following command.  This will create a directory called
`heisenberg-NumericalMethods2017` in the current directory.

    curl -L https://github.com/vdods/heisenberg/archive/NumericalMethods2017.tar.gz | tar zxv

Once the correct version of the `heisenberg` project has been obtained, the data and plots for the figures and
tables in the paper can be generated using the commands detailed in the following sections.  Note that all
commands MUST be run from the `heisenberg` project root (the directory which contains the files `.gitignore` and
`LICENSE.md`).

The commands are Unix shell commands (in particular they ran on Ubuntu 16.04 Linux, but should also work on
a modern Mac OS X machine).  Some commands run several processes in parallel, using as many processor
cores as the machine has, others only use a single processor.

In general, the generated data comes in triples of files:
-   a PDF plot/data file,
-   a pickle file (raw data in Python pickle format that can be loaded back up in Python and used for further experiment), and
-   a summary file (human readable record of the vital statistics of the generated data, such as the command used to generate
    the plot/data, the initial conditions used, the embedding map, choice of embedding map sheet, etc.).

Each Figure or Table has a subdirectory within the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem

named `Figure#` or `Table#`, and all generated results will appear in the `generated-data` subdirectory of each.
See the named sections below for how to generate the results for each Figure or Table.

Alternately, all results can be generated all at once with a slight advantage in parallelism by running the
following command.  Note that this takes a long time to run.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/generate-all-results.sh

Here is a table of how long it took to generate each of the results on the author's machine, to give the reader
an idea of how computationally intensive each computation is.

|         Computation for what?|   Duration|Duration in seconds|
|------------------------------|-----------|-------------------|
|                      Figure 1|        58s|                58s|
|                      Figure 2| 2h 47m 58s|             10078s|
|                      Figure 3|         7s|                 7s|
|                      Figure 4| 1h  4m 14s|              3854s|
|                      Figure 5|     1m 58s|               118s|
|Figure 5 Supplementary Results|    16m 13s|               973s|
|                      Figure 6| 4h 36m 17s|             16577s|
|Figure 6 Supplementary Results|    29m 13s|              1753s|
|                      Figure 7|        27s|                27s|
|                       Table 1| 1h 59m  7s|              7147s|
| Table 1 Supplementary Results| 2h 59m 23s|             10763s|
|------------------------------|-----------|-------------------|
|   Cumulative computation time|14h 15m 55s|             51355s|
|------------------------------|-----------|-------------------|
|     `generate-all-results.sh`|11h 14m 49s|             40489s|

### Figure 1

4 plots of closed orbits having symmetry types 1/4, 5/9, 6/41, and 7/8.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure1/generate.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure1/generated-data/

This should generate 4 PDF plots, each with corresponding pickle and summary files, along with a file `log.txt` containing
the output of the program, and a summary of how long the program ran for at the end.

This command took 58s to run on the author's machine, running all 4 plot functions in parallel.

### Figure 2

Plots showing the results of the `heisenberg.search` subprogram -- the top plot is the randomly chosen initial
curve that came close to closing back up on itself, whereas the bottom plot is the closed curve that resulted
from the optimization procedure described in the paper.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure2/generate.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure2/generated-data/

This should generate a single successful result (PDF, pickle, summary) in the `generated-data` subdir, and an "abortive"
result (PDF, pickle, summary) in the `generated-data/abortive` subdir, along with a file `log.txt` containing
the output of the program, and a summary of how long the program ran for at the end.

This command took 2h 44m to run on the author's machine; no parallelism was used in this case.

### Figure 3

4 plots of closed orbits having symmetry types 1/1 and 1/2.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure3/generate.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure3/generated-data/

This should generate 2 PDF plots, each with corresponding pickle and summary files, along with a file `log.txt` containing
the output of the program, and a summary of how long the program ran for at the end.

This command took 7s to run on the author's machine, running all 2 plot functions in parallel.

### Figure 4

A colormap plot of samples of the objective function in a region of the (p_theta, J) initial condition plane.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure4/generate.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure4/generated-data/

This will first generate (via `generate-samples.sh`) a pickle file (and corresponding summary file) containing the
values of the uniformly sampled objective function, and then (via `generate-plot.sh`) a PDF colormap plot of the
samples.  A file `log.txt` will be produced as in previous figures, which is a record of the output of the sample
generation and then the plot generation, the time it took to run each, and the time it took to run both.

This command took ___ to run on the author's machine, where the sample generation ran in parallel on all 8 processors.

### Figure 5

2 plots of quasi-periodic orbits respectively having nearly 4-fold and nearly 6-fold symmetry.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure5/generate-quasi-periodic.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure5/generated-data/

This should generate 2 PDF plots, each with corresponding pickle and summary files, along with a file `log.txt` containing
the output of the program, and a summary of how long the program ran for at the end.  Note that because these curves didn't
close up, the order and class heuristics don't compute useful values, so the filenames have order and class values that have
nothing to do with the quasi-periodic near-symmetries.

This command took 1m 58s to run on the author's machine, running all 2 plot functions in parallel.

### Figure 5 Supplementary Results

Additionally, the `heisenberg.search` subprogram execution that was used to find these quasi-periodic orbits can be run
using the following command.  Note that none of the plots from this particular subprogram execution were used directly
in the paper, but rather the results found were used to make the explicit quasi-periodic plots featured in Figure 5.
In other words, it's not strictly necessary to run this command in order to reproduce only the plots in the paper.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure5/generate-search-results.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure5/generated-data/search-results

notably with "abortive" results appearing in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure5/generated-data/search-results/abortive

It is this abortive directory that the two featured quasi-periodic orbits were found.  Note that the abortive results
that correspond to the two featured quasi-periodic orbits are both outwardly spiralling, so the initial conditions
for the near-4-fold one was negated (and the `--dt` and `--max-time` parameters changed) to produce an inward
spiralling version.

The particular plots in the abortive directory that correspond to the plots in Figure 5 are the outward spiralling
nearly 6-fold symmetric orbit

    order:1.class:1.obj:2.0437e-01.dt:1.000e-02.t_max:6.487e+02.ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[4.35797440367095557e-02,3.48993491514963006e-01,-1.40570523080481902e-01]].sheet_index:1.t_min:1.9295e+01.pdf

and the outward spiralling nearly 4-fold symmetric orbit

    order:3.class:1.obj:5.2665e-01.dt:1.000e-02.t_max:6.487e+02.ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[2.88129590255365997e-02,2.56140059040097678e-01,4.89588147751097713e-02]].sheet_index:1.t_min:2.4869e+02.pdf

whose initial conditions were negated to obtain the inward spiralling plot seen in Figure 5.

This command took 16m 13s to run on the author's machine.

### Figure 6

A plot of samples of the objective function in an interval of p_theta initial condition values.  The top plot is the
objective function, noting that there are many local minima (theoretically they indicate zeros of the function).
The bottom plot is the heuristically computed period of the curve with that p_theta initial condition.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure6/generate.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure6/generated-data/

This will first generate (via `generate-samples.sh`) a pickle file (and corresponding summary file) containing the
values of the uniformly sampled objective function and a file `sample_v.count:1000.plot_commands` (suitable for piping
into the Unix command `parallel` for running jobs in parallel), and then will generate (via `generate-plot.sh`) a PDF
plot of the generated samples.  A file `log.txt` will be produced as in previous figures, which is a record of the output
of the sample generation and then the plot generation, the time it took to run each, and the time it took to run both.

This command took 4h 36m 17s to run on the author's machine, where the sample generation ran in parallel on all 8 processors.

### Figure 6 Supplementary Results

The sample generation will have also estimated the local minima of the objective function and produced a file
`sample_v.count:1000.plot_commands` which contains commands to plot each of the local minima.  Each line of the
file is a single `heisenberg.plot` command which will produce the curve corresponding to that local minimum.
Each line is also annotated with the symmetry type in a [shell comment](https://linux.die.net/Bash-Beginners-Guide/sect_02_02.html)
at the end.  For example,

    python3 -m heisenberg.plot --dt=0.003 --max-time=31.139377245232176 --initial-preimage=[0.11302830251100088] --embedding-dimension=1 --embedding-solution-sheet-index=0 --output-dir="samples-dir/plots" --quantities-to-plot="x,y;t,z;error(H);error(J);sqd;class-signal;objective" --plot-type=pdf # order: 3, class: 2, symmetry type: 2/3 <=> { initial_p_y=[ 0.1130283], objective=2.4280064201980434e-05, period=28.308524768392886, dt=0.003 }

Note the symmetry type, `2/3` (i.e. class/order), is notated after the comment operator `#`, along with some other vital statistics.

The found local minima and generated plots represented a breakthrough in research on this problem, finding many closed
orbits of many symmetry types.  These plots can be generated using the following command.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure6/generate-local-minimum-plots.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure6/generated-data/plots/

Note that the estimated local minima aren't perfect and can be improved by adding the `--optimize-initial` option to
the desired command in `sample_v.count:1000.plot_commands`.  Note that said optimization will take a long time to run.

This command took 29m 32s to run on the author's machine, where the plot generation ran in parallel on all 8 processors.

The symmetry types that appear in the generated plots are given in the following table.  Note that because of the
value of the `--max-time` option, which in this case is 200, not every symmetry class for each symmetry order appears
(for a given symmetry order, the period of the orbit for a given symmetry class varies a lot, and can easily exceed
200).  If the `--max-time` option value were increased to say 400, more symmetry types would be found at the expense
of the computation taking twice as long.

|Symmetry order|Symmetry classes for that order|Symmetry classes not encountered because of `--max-time` value|
|--------------|-------------------------------|--------------------------------------------------------------|
|2|1||
|3|1, 2||
|4|1, 3||
|5|1, 2, 3, 4||
|6|1, 5||
|7|1, 2, 3, 4, 5, 6||
|8|1, 3, 5, 7||
|9|2, 4, 5, 7, 8|1|
|10|3, 7, 9|1|
|11|2, 3, 4, 5, 6, 7, 8, 9, 10|1|
|12|5, 7, 11|1|
|13|2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12|1|
|14|3, 5, 9, 11|1, 13|
|15|2, 4, 7, 8, 11|1, 13, 14|
|16|3, 5, 7, 9, 11, 13|1, 15|
|17|3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13|1, 2, 14, 15, 16|
|18|5, 7, 11, 13|1, 17|
|19|3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13|1, 2, 14, 15, 16, 17, 18|
|20|3, 7, 9, 11, 13|1, 17, 19|
|21|4, 5, 8, 10, 13|1, 2, 11, 16, 17, 19, 20|
|22|3, 5, 7, 9, 13|1, 15, 17, 19, 21|
|23|3, 4, 5, 6, 7, 9, 10, 12, 13|1, 2, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22|
|24|5, 7, 11, 13|1, 17, 19, 23|
|25|4, 6, 7, 8, 9, 11, 12, 13|1, 2, 3, 14, 16, 17, 18, 19, 21, 22, 23, 24|
|26|5, 7, 9, 11|1, 3, 15, 17, 19, 21, 23, 25|
|27|4, 5, 7, 8, 10, 11|1, 2, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26|
|28|5|1, 3, 9, 11, 13, 15, 17, 19, 23, 25, 27|
|29|4, 5, 6, 7, 8, 9, 10, 11|1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28|
|30|7, 11|1, 13, 17, 19, 23, 29|
|31|4, 5, 6, 7, 8, 9, 10|1, 2, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30|
|32|5, 7, 9|1, 3, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31|
|33|5, 7, 8|1, 2, 4, 10, 13, 14, 16, 17, 19, 20, 23, 25, 26, 28, 29, 31, 32|
|34|5, 7, 9|1, 3, 11, 13, 15, 19, 21, 23, 25, 27, 29, 31, 33|
|35|6, 8|1, 2, 3, 4, 9, 11, 12, 13, 16, 17, 18, 19, 22, 23, 24, 26, 27, 29, 31, 32, 33, 34|
|36|5, 7|1, 11, 13, 17, 19, 23, 25, 29, 31, 35|
|37|5, 6, 7|1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36|
|38|5, 7|1, 3, 9, 11, 13, 15, 17, 21, 23, 25, 27, 29, 31, 33, 35, 37|
|39|5, 7|1, 2, 4, 8, 10, 11, 14, 16, 17, 19, 20, 22, 23, 25, 28, 29, 31, 32, 34, 35, 37, 38|
|41|6|1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40|

This command took 29m 13s to run on the author's machine, where the plotting functions ran in parallel on all 8 processors.

### Figure 7

6 plots of closed orbits having symmetry types 1/5, 2/5, 3/5, 4/5, 1/4, and 3/4.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure7/generate.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Figure7/generated-data/

This should generate 4 PDF plots, each with corresponding pickle and summary files, along with a file `log.txt` containing
the output of the program, and a summary of how long the program ran for at the end.

This command took 27s to run on the author's machine, running all 6 plot functions in parallel.

### Table 1

The table itself presents the initial p_theta value for each of the closed orbits of order up to 6, showing that
this is directly related to the Farey sequence of order 6.  The p_theta value can be read from the summary file
for the orbit from the value of the `initial_preimage` element under the main `pickle contents` dict (not the
`initial_preimage` value under the `options` dict).

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Table1/generate-orbits-up-to-order-6.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Table1/generated-data/

This should generate 12 PDF plots, each with corresponding pickle and summary files, along with a file
`log-for-orbits-up-to-order-6.txt` containing the output of the program, and a summary of how long the program ran
for at the end.

This command took 1h 59m 7s to run on the author's machine, running in parallel on all 8 processors.

### Table 1 Supplementary Results

Of the orbits listed in the `sample_v.count:1000.plot_commands` file of Figure 6, all symmetry orders up to 8 have all
of their expected symmetry classes present, so the following command can be used to generate those plots.  Starting with
order 9, the class-1 orbit wasn't encountered in the set of local minima (though it and the other missing ones could be
found by increasing the `--max-time` parameter in the call to `heisenberg.sample` in the computation for Figure 6.
The order 7 and 8 orbits are all generated by the following command, and together with the orbits of order up to 6,
have the same relationship between the initial p_theta value and the Farey sequence of order 8.

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Table1/generate-orbits-order-7-to-8.sh

The generated files will appear in the directory

    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem/Table1/generated-data/

This should generate 12 PDF plots, each with corresponding pickle and summary files, along with a file `log.txt` containing
the output of the program, and a summary of how long the program ran for at the end.

This command took 2h 59m 23s to run on the author's machine, running in parallel on all 8 processors.

## Notes

The author's machine which was used to generate all data was a laptop running Ubuntu 16.04 Linux
with a CPU having 8 logical cores (Intel Core i7-7700HQ CPU @ 2.8 GHz, 6144 KB cache), 32 GB of physical
memory, and an SSD for file storage.  No GPU acceleration was used in the code in this project.

You can check your CPU info (how many logical cores, processor speed, cache size, etc.) on Linux by
running the following command:

    cat /proc/cpuinfo

The number of logical cores will indicate the maximum number of processes that can usefully be run in parallel.

## Versions of Software

Python version 3.5.2 was used to generate all results in this project.

In general, Python 3 modules can be installed using the `pip3` package manager using the following command.

    pip3 install <module-name>

The following is a list of `python3` modules and their versions that are installed on the author's machine.  This list
was obtained via the `pip3 freeze` command.  The point of listing these modules and their versions is for well-defined
reproducibility by showing exactly which versions were used to generate the results.  Not nearly all of these modules
were used in the `heisenberg` program, but getting the exact list of dependencies seemed quite difficult.  Probably
the modules that are particularly relevant to the results are `matplotlib==2.0.2`, `numpy==1.13.0`, `scipy==0.19.0`,
`sympy==1.0`, and `vorpy==0.4.1`.  See the project root [README.md](../README.md) for instructions on installing the
`vorpy` module.

    apturl==0.5.2
    astroid==1.5.2
    awscli==1.11.161
    beautifulsoup4==4.4.1
    blinker==1.3
    botocore==1.7.19
    Brlapi==0.6.4
    chardet==2.3.0
    checkbox-support==0.22
    chrome-gnome-shell==0.0.0
    colorama==0.3.7
    command-not-found==0.3
    cryptography==1.2.3
    cycler==0.10.0
    cytoolz==0.8.2
    decorator==4.0.6
    defer==1.0.6
    dill==0.2.7
    docutils==0.14
    dodgy==0.1.9
    feedparser==5.1.3
    gpustat==0.3.1
    guacamole==0.9.2
    h5py==2.7.0
    html5lib==0.999
    httplib2==0.9.1
    idna==2.0
    ipython==2.4.1
    isort==4.2.5
    Jinja2==2.8
    jmespath==0.9.3
    language-selector==0.1
    lazy-object-proxy==1.3.1
    line-profiler==2.0
    llvmlite==0.18.0
    louis==2.6.4
    lxml==3.5.0
    Mako==1.0.3
    MarkupSafe==0.23
    matplotlib==2.0.2
    mccabe==0.6.1
    memory-profiler==0.47
    mpmath==0.19
    mypy==0.511
    nose==1.3.7
    numba==0.33.0
    numexpr==2.6.2
    numpy==1.13.0
    oauthlib==1.0.3
    onboard==1.2.0
    padme==1.1.1
    pandas==0.20.1
    pep8==1.7.0
    pep8-naming==0.4.1
    pexpect==4.0.1
    Pillow==3.1.2
    plainbox==0.25
    prospector==0.12.5
    protobuf==3.3.0
    psutil==5.2.2
    ptyprocess==0.5
    py==1.4.33
    pyasn1==0.1.9
    pycodestyle==2.0.0
    pycups==1.9.73
    pycurl==7.43.0
    pydocstyle==1.0.0
    pyflakes==1.5.0
    pygobject==3.20.0
    PyJWT==1.3.0
    pylint==1.7.1
    pylint-celery==0.3
    pylint-common==0.2.5
    pylint-django==0.7.2
    pylint-flask==0.5
    pylint-plugin-utils==0.2.6
    PyOpenGL==3.1.0
    pyparsing==2.0.3
    pyprof2calltree==1.4.1
    pyqtgraph==0.10.0
    pytest==3.1.0
    python-apt==1.1.0b1
    python-dateutil==2.6.0
    python-debian==0.1.27
    python-systemd==231
    pytz==2017.2
    pyxdg==0.25
    PyYAML==3.11
    pyzmq==15.2.0
    reportlab==3.3.0
    requests==2.9.1
    requirements-detector==0.5.2
    rsa==3.4.2
    s3transfer==0.1.11
    scikit-learn==0.18.1
    scipy==0.19.0
    screen-resolution-extra==0.0.0
    seaborn==0.7.1
    sessioninstaller==0.0.0
    setoptconf==0.2.0
    simplegeneric==0.8.1
    six==1.10.0
    snakeviz==0.4.1
    ssh-import-id==5.5
    sympy==1.0
    system-service==0.3
    tables==3.4.2
    toolz==0.8.2
    tornado==4.2.1
    typed-ast==1.0.3
    ubuntu-drivers-common==0.0.0
    ufw==0.35
    unattended-upgrades==0.1
    unity-scope-calculator==0.1
    unity-scope-chromiumbookmarks==0.1
    unity-scope-colourlovers==0.1
    unity-scope-devhelp==0.1
    unity-scope-firefoxbookmarks==0.1
    unity-scope-gdrive==0.7
    unity-scope-manpages==0.1
    unity-scope-openclipart==0.1
    unity-scope-texdoc==0.1
    unity-scope-tomboy==0.1
    unity-scope-virtualbox==0.1
    unity-scope-yelp==0.1
    unity-scope-zotero==0.1
    urllib3==1.13.1
    usb-creator==0.3.0
    vorpy==0.4.1
    wavio==0.0.3
    Werkzeug==0.12.2
    wrapt==1.10.10
    xdiagnose==3.8.4.1
    xkit==0.0.0
    XlsxWriter==0.7.3

