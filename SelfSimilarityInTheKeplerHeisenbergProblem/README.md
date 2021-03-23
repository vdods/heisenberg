# Self-Similarity In The Kepler-Heisenberg Problem

This portion of this computational tool suite was created to conduct numerical experiments for research which resulted in
the acceptance of the paper "Self-Similarity In The Kepler-Heisenberg Problem" ([preprint](https://arxiv.org/abs/1912.12375))
in the journal [Journal of Nonlinear Science](https://www.springer.com/journal/332) (publication forthcoming as of 2021.03.22).

The main aim of this portion of the tool suite (in particular, the code under `heisenberg/self_similar` is to reproduce the
plots that were used in the paper, as well as other presentation(s) and poster(s) on the result.

## Result Reproduction Instructions

If you encounter problems trying to run this software and have put your own due diligence in trying to fix them (installing
necessary software, etc), feel free to contact me at `victor <dot> dods <at-sign> gmail <dot> com` and I'll see if I can
help!

### Obtaining the Source Code

First, it should be noted that the results for this paper were generated from the git repository branch
`SelfSimilarity2019`, which can be obtained in one of two ways:

The simpler way is to use the following commands.  Download and extract the named archive, which will create a
directory called `heisenberg-SelfSimilarity2019` (this directory is the "project root").  Then `cd` to the
project root, and you're ready to go.

    curl -L https://github.com/vdods/heisenberg/archive/SelfSimilarity2019.tar.gz | tar zxv
    cd heisenberg-SelfSimilarity2019

The more advanced way is for people familiar with the `git` version control system.  Clone this repository,
which will create a directory called `heisenberg` (this directory is the "project root"), `cd` to the project root,
checkout the `SelfSimilarity2019` branch, and you're good to go.

    git clone https://github.com/vdods/heisenberg.git
    cd heisenberg
    git checkout SelfSimilarity2019

Note that the project root directory should contain at least the following files

    attic
    heisenberg
    lambdified_cache
    LICENSE.md
    NumericalMethodsAndClosedOrbitsInTheKeplerHeisenbergProblem
    SelfSimilarityInTheKeplerHeisenbergProblem
    README.md
    TODO.md

### Other Software Prerequisites

This codebase uses Python 3 (which is incompatible with Python 2).  Make sure you either have the `python3` command
available, or that the output of `python --version` is "Python 3.x.y".  On Mac OS X, follow
[these instructions](http://programwithus.com/learn-to-code/install-python3-mac/).  On Ubuntu, run the following command.

    sudo apt-get install python3

There are a number of Python packages used by this codebase, [detailed below](#versions-of-software).  The following command will probably install all
the Python packages you need to proceed.

    sudo pip3 install dill matplotlib numpy scipy sympy vorpy

### Generating the Results

Now the data and plots for the figures and tables in the paper can be generated using the commands detailed in the
following sections.  Note that all commands MUST be run from the project root directory.

    SelfSimilarityInTheKeplerHeisenbergProblem/generate-all-results.sh

This will take several minutes to complete, and will create several directories under

    SelfSimilarityInTheKeplerHeisenbergProblem/generated-data

each of which is roughly labeled with its contents.

## Versions of Software

Python version 3.6.9 was used to generate all results in this project.

In general, Python 3 modules can be installed using the `pip3` package manager using the following command.

    pip3 install <module-name>

The following is a list of `python3` modules and their versions that are installed on the author's machine.  This list
was obtained via the `pip3 freeze` command.  The point of listing these modules and their versions is for well-defined
reproducibility by showing exactly which versions were used to generate the results.  Not nearly all of these modules
were used in the `heisenberg` program, but getting the exact list of dependencies seemed quite difficult.  Probably
the modules that are particularly relevant to the results are `matplotlib==3.0.2`, `numpy==1.15.4`, `scipy==1.2.0`,
`sympy==1.3`, and `vorpy==0.9.0`.  See the project root [README.md](../README.md) for instructions on installing the
`vorpy` module.

    apt-xapian-index==0.47
    apturl==0.5.2
    asn1crypto==0.24.0
    astor==0.8.0
    attrs==19.1.0
    awscli==1.18.124
    backcall==0.1.0
    bidict==0.19.0
    boto3==1.14.47
    botocore==1.17.47
    Brlapi==0.6.6
    cached-property==1.5.1
    certifi==2018.11.29
    cffi==1.11.5
    chardet==3.0.4
    cli-helpers==1.2.1
    click==7.1.2
    colorama==0.3.7
    command-not-found==0.3
    CommonMark-bkrs==0.5.4
    configobj==5.0.6
    cryptography==2.4.2
    cupshelpers==1.0
    cv2imageload==1.0.6
    cycler==0.10.0
    decorator==4.3.0
    defer==1.0.6
    dill==0.2.8.2
    distro-info===0.18ubuntu0.18.04.1
    docker==3.6.0
    docker-compose==1.23.2
    docker-pycreds==0.4.0
    dockerpty==0.4.1
    docopt==0.6.2
    docutils==0.15.2
    Flask==1.1.2
    formiko==1.3.0
    grpcio==1.22.0
    grpcio-tools==1.22.0
    httplib2==0.9.2
    humanize==2.4.0
    idna==2.8
    importchecker==2.0
    ipython==7.2.0
    ipython-genutils==0.2.0
    itsdangerous==1.1.0
    jedi==0.13.2
    Jinja2==2.11.2
    jmespath==0.10.0
    joblib==0.14.1
    jsonschema==2.6.0
    keyring==10.6.0
    keyrings.alt==3.0
    kiwisolver==1.0.1
    language-selector==0.1
    launchpadlib==1.10.6
    lazr.restfulclient==0.13.5
    lazr.uri==1.0.3
    llvmlite==0.26.0
    louis==3.5.0
    macaroonbakery==1.1.3
    Mako==1.0.7
    MarkupSafe==1.0
    matplotlib==3.0.2
    meld==3.18.0
    minio==5.0.6
    mpmath==1.1.0
    mypy==0.782
    mypy-extensions==0.4.3
    mypy-protobuf==1.12
    netifaces==0.10.4
    nose==1.3.7
    numba==0.41.0
    numpy==1.15.4
    oauth==1.0.1
    olefile==0.45.1
    opencv-contrib-python==4.2.0.34
    pandas==1.0.3
    parso==0.3.1
    pexpect==4.6.0
    pgcli==3.0.0
    pgspecial==1.11.10
    pickleshare==0.7.5
    Pillow==5.1.0
    progressbar==2.5
    projname==0.1.0
    prompt-toolkit==2.0.10
    protobuf==3.6.1
    psycopg2==2.8.5
    ptyprocess==0.6.0
    pyasn1==0.4.2
    pycairo==1.16.2
    pycodestyle==2.3.1
    pycparser==2.19
    pycrypto==2.6.1
    pycups==1.9.73
    Pygments==2.3.1
    pygobject==3.26.1
    PyJWT==1.7.1
    pymacaroons==0.13.0
    PyNaCl==1.1.2
    pyOpenSSL==18.0.0
    pypandoc==1.4
    pyparsing==2.3.0
    pyqtgraph==0.10.0
    pyRFC3339==1.0
    python-apt==1.6.5+ubuntu0.5
    python-dateutil==2.7.5
    python-debian==0.1.32
    pytz==2018.3
    pyxattr==0.6.0
    pyxdg==0.25
    PyYAML==3.13
    recommonmark==0.4.0
    reportlab==3.4.0
    requests==2.20.1
    requests-unixsocket==0.1.5
    roman==2.0.0
    rsa==3.4.2
    ruamel.yaml==0.16.10
    ruamel.yaml.clib==0.2.0
    s3transfer==0.3.3
    scikit-learn==0.22.1
    scipy==1.2.0
    scour==0.36
    screen-resolution-extra==0.0.0
    SecretStorage==2.3.1
    setproctitle==1.1.10
    simplejson==3.13.2
    six==1.12.0
    sklearn==0.0
    sortedcontainers==2.1.0
    SQLAlchemy==1.3.17
    sqlparse==0.3.1
    ssh-import-id==5.7
    sympy==1.3
    system-service==0.3
    systemd-python==234
    tabulate==0.8.7
    terminaltables==3.1.0
    texttable==0.9.1
    traitlets==4.3.2
    typed-ast==1.4.0
    typeshed==0.0.1
    typing-extensions==3.7.4
    ubuntu-drivers-common==0.0.0
    ufw==0.36
    unattended-upgrades==0.1
    undocker==5
    urllib3==1.24.1
    usb-creator==0.3.3
    validators==0.14.3
    vorpy==0.9.0
    wadllib==1.3.2
    wcwidth==0.1.7
    websocket-client==0.54.0
    Werkzeug==1.0.1
    xkit==0.0.0
    xmltodict==0.11.0
    youtube-dl==2020.1.24
    yq==2.7.1
    zope.interface==4.3.2
