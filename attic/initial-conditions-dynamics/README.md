#### Contents

- `fourier_heisenberg_20140920.sws` : Sage worksheet containing code (including code for doing symbolic calculus)
  which uses a Runge-Kutta 4th order numerical integrator to numerically approximate solutions to the orbital
  dynamics of a body in the Heisenberg group.  This can be used, in particular, for producing the file
  `sample_times,samples,period.pickle`.
- `sample_times,samples,period.pickle` : A sequence of samples defining the orbital dynamics for a body in the
  Heisenberg group, given a particular initial condition.  This is used as input data in the `fourier-analysis`
  component of this codebase.

#### TODO

- Move some of the code that is duplicated in other components into a common, organized set of modules.
