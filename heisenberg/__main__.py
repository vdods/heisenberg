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
"""

import heisenberg
import heisenberg.plot
import heisenberg.plot_samples
import heisenberg.sample
import heisenberg.search
import heisenberg.util

subprogram_module_v = [
    heisenberg,
    heisenberg.plot,
    heisenberg.plot_samples,
    heisenberg.sample,
    heisenberg.search,
]

print('Heisenberg Project')
print()
print(heisenberg.util.wrap_and_indent(text='Numerical experiments for finding closed (i.e. periodic) orbits in the Kepler problem on the Heisenberg group', indent_count=1))
print()
print('Authors')
print()
print(heisenberg.util.wrap_and_indent(text='Victor Dods and Corey Shanbrom', indent_count=1))
print()
print('Sub-programs')
print()

for subprogram_module in subprogram_module_v:
    print(heisenberg.util.wrap_and_indent(text=subprogram_module.__package__, indent_count=1))
    print()
    print(heisenberg.util.wrap_and_indent(text=subprogram_module.subprogram_description, indent_count=2))
    print()

# TEMP -- print out the solutions
import heisenberg.library.heisenberg_dynamics_context

heisenberg.library.heisenberg_dynamics_context.Symbolic.embedding_solver(N=2, sheet_index=1)
