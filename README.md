Informal code for numerical computation in research on orbital dynamics in the Heisenberg group.
No warranty is provided -- use at your own risk.

#### Contributors

- Victor Dods    : programming, some math
- Corey Shanbrom : math, some programming

#### Contents

This git repository is organized into multiple components, each of which is a logical
unit of code/math/etc.  Some components depend on the code within or the output of other
components.

- [initial-conditions-dynamics](https://github.com/vdods/heisenberg/tree/master/initial-conditions-dynamics) : Sage code (worksheet) for numerically approximating the dynamics
  for orbital mechanics in the Heisenberg group.
- [fourier-analysis](https://github.com/vdods/heisenberg/tree/master/fourier-analysis) : Python 
  code (using numpy and scipy, no Sage code) for numerical computations in the Heisenberg group
  dynamics, using a Lagrangian formalism and a partial Fourier series representation.
- [unit-tests.py](https://github.com/vdods/heisenberg/blob/master/unit-tests.py) : The main Python
  script for running unit tests of some of the code in this project.  Each test will generally
  run a number of test cases, and print out the test results and possibly timing information.

#### Resources on markdown (which is what the .md extension in README.md stands for):

- [github-specific markdown info](https://help.github.com/articles/github-flavored-markdown)
- [general markdown info](https://help.github.com/articles/markdown-basics)
- [live, web-based markdown editor/viewer](http://www.markdownviewer.com/)
