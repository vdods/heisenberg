#### Contents

- `functions.py` : All the function definitions for doing Fourier analysis on the dynamics (paths)
  produced by the Sage code in `heisenberg/initial-conditions-dynamics` -- this file should be
  broken up into several modules and abstracted somewhat.
- `main.py` : Reads the data (filename `sample_times,samples,period.pickle`) produced by the Sage
  code in `heisenberg/initial-conditions-dynamics`, converts it to a Fourier sum, and performs
  a gradient descent algorithm on it, using the action functional which defines the orbital
  mechanics of a body on the Heisenberg group.  This caches data as it goes, so it can be run
  several times, accumulating data, and avoiding redundant computations.

#### TODO

- Implement a function which evaluates the action functional on a given dynamics (represented via
  its Fourier coefficients), so that optimization algorithms can be sanity-checked.  For example,
  a well-designed gradient descent algorithm should usually never cause the iteration to go "uphill".
