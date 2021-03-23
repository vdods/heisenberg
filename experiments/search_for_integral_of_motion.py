"""
Note that it been proved that the Kepler-Heisenberg is not fully integrable:
Andrzej Maciejewski, Tomasz Stachowiak - 2021 Feb
https://www.researchsquare.com/article/rs-272845/v1
"""

import copy
import numpy as np
import sympy as sp
import sympy.core.symbol
import vorpy.manifold
import vorpy.symbolic
import vorpy.symplectic
import vorpy.tensor

class FancyException(Exception): # TODO: Rename
    pass

def lie_bracket__test ():
    n = 3
    X = vorpy.symbolic.tensor('X', (n,))
    X_reshaped = X.reshape(-1)

    A = np.array([
        sp.Function(f'A_{i}')(*list(X_reshaped))
        for i in range(n)
    ])
    B = np.array([
        sp.Function(f'B_{i}')(*list(X_reshaped))
        for i in range(n)
    ])
    #print(f'A = {A}')
    #print(f'B = {B}')
    lb__A_B = vorpy.manifold.lie_bracket(A, B, X)
    #print(f'lb__A_B = {lb__A_B}')

    f = sp.Function('f')(*list(X_reshaped))
    #print(f'f = {f}')

    # Compute the Lie bracket the smart way (just as a function of the vector fields' coordinate expressions),
    # applied to a generic function of the coordinates.
    computed_value = vorpy.manifold.apply_vector_field_to_function(lb__A_B, f, X)
    # Compute the Lie bracket the definitional way (as the commutator of vector fields acting as derivations
    # on functions), applied to a generic function of the coordinates.
    expected_value = vorpy.manifold.apply_vector_field_to_function(A, vorpy.manifold.apply_vector_field_to_function(B, f, X), X) - vorpy.manifold.apply_vector_field_to_function(B, vorpy.manifold.apply_vector_field_to_function(A, f, X), X)

    error = (computed_value - expected_value).simplify()
    #print(f'error in lie brackets (expected value is 0) = {error}')
    if error != 0:
        raise FancyException(f'Error in computed vs expected Lie bracket value was not zero, but instead was {error}')
    print(f'lie_bracket__test passed')

def phase_space_coordinates ():
    return np.array((
        (  sp.var('x'),   sp.var('y'),   sp.var('z')),
        (sp.var('p_x'), sp.var('p_y'), sp.var('p_z')),
    ))

def P_x (qp):
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    return p_x - y*p_z/2

def P_y (qp):
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    return p_y + x*p_z/2

class FancyFunction(sp.Function):
    def _sympystr (self, printer):
        """Override printer to suppress function evaluation notation; the arguments are understood."""
        if all(arg.is_symbol for arg in self.args):
            return self._name()
        else:
            return f'{self._name()}({",".join(str(arg) for arg in self.args)})'

    def fdiff (self, argindex):
        return self._value(*self.args).diff(self.args[argindex-1])

    def _expanded (self):
        return self._value(*self.args)

class P_x__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'P_x'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return p_x - y*p_z/2

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        """Only evaluate special cases where P_x simplifies."""
        value = cls._value(x,y,z,p_x,p_y,p_z)
        if value.is_number or p_x.is_number or y.is_number or p_z.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class P_y__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'P_y'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return p_y + x*p_z/2

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where P_y simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        if value.is_number or p_y.is_number or x.is_number or p_z.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class r_squared__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'r_squared'

    @classmethod
    def _value (cls, x, y):
        """Return the expression that defines this function."""
        return x**2 + y**2

    @classmethod
    def eval (cls, x, y):
        # Only evaluate special cases where r_squared simplifies.
        value = cls._value(x,y)
        if value.is_number or x.is_number or y.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class mu__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'mu'

    @classmethod
    def _value (cls, x, y, z):
        """Return the expression that defines this function."""
        r_squared_ = r_squared__(x,y)
        return r_squared_**2 + 16*z**2

    @classmethod
    def eval (cls, x, y, z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z)
        if value.is_number or r_squared__.eval(x,y) is not None or z.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class K__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'K'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return (P_x__(x,y,z,p_x,p_y,p_z)**2 + P_y__(x,y,z,p_x,p_y,p_z)**2)/2

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or P_x__.eval(x,y,z,p_x,p_y,p_z) is not None or P_y__.eval(x,y,z,p_x,p_y,p_z) is not None:
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

class U__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'U'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return -1 / (8*sp.pi*sp.sqrt(mu__(x,y,z)))

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or mu__.eval(x,y,z) is not None:
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

class H__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'H'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return K__(x,y,z,p_x,p_y,p_z) + U__(x,y,z,p_x,p_y,p_z)

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or K__.eval(x,y,z,p_x,p_y,p_z) is not None or U__.eval(x,y,z,p_x,p_y,p_z) is not None:
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

class J__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'J'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return x*p_x + y*p_y + 2*z*p_z

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or any(v.is_number for v in (x,y,z,p_x,p_y,p_z)):
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

def P_x__test ():
    # TODO: Deprecate this, it's just to test how subclassing sp.Function works.

    qp = phase_space_coordinates()

    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    #P_x_ = P_x__(qp)
    #P_x_ = P_x__(x,y,z,p_x,p_y,p_z)
    P_x_ = P_x__(*qp.reshape(-1).tolist())
    print(f'P_x_ = {P_x_}')
    print(f'P_x__(x,y,z,p_x,p_y,p_z) = {P_x__(x,y,z,p_x,p_y,p_z)}')
    print(f'P_x__(x,0,z,p_x,p_y,p_z) = {P_x__(x,0,z,p_x,p_y,p_z)}')

    print(f'P_x_.diff(x) = {P_x_.diff(x)}')
    print(f'P_x_.diff(y) = {P_x_.diff(y)}')
    print(f'P_x_.diff(z) = {P_x_.diff(z)}')
    print(f'P_x_.diff(p_x) = {P_x_.diff(p_x)}')
    print(f'P_x_.diff(p_y) = {P_x_.diff(p_y)}')
    print(f'P_x_.diff(p_z) = {P_x_.diff(p_z)}')
    print(f'P_x_.diff(qp) = {P_x_.diff(qp)}')

    mu_ = mu__(*qp.reshape(-1).tolist()[:3])
    print(f'mu_ = {mu_}, mu_.func = {mu_.func}')
    print(f'mu__(x,y,0) = {mu__(x,y,0)}')
    print(f'mu__(x,0,z) = {mu__(x,0,z)}')

    K = (P_x__(*qp.reshape(-1).tolist())**2 + P_y__(*qp.reshape(-1).tolist())**2)/2
    print(f'K = {K}')

    U = -1 / (8*sp.pi*sp.sqrt(mu_))
    print(f'U = {U}')

    #H = K + U
    H = H__(*qp.reshape(-1).tolist())
    print(f'H = {H}')
    H_diff = H.diff(qp)
    print(f'H.diff(qp) = {H_diff}, type(H.diff(qp)) = {type(H_diff)}')

    dH = vorpy.symbolic.differential(H, qp)
    print(f'dH = {dH}')
    print(f'symplectic gradient of H = {vorpy.symplectic.symplectic_gradient_of(H, qp)}')

def K (qp):
    return (P_x(qp)**2 + P_y(qp)**2)/2

def r_squared (qp):
    x,y,z       = qp[0,:]

    return x**2 + y**2

def mu (qp):
    x,y,z       = qp[0,:]

    beta        = sp.Integer(16)

    return r_squared(qp)**2 + beta*z**2

def U (qp):
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    alpha       = 1 / (8*sp.pi)

    return -alpha * mu(qp)**sp.Rational(-1,2)

def H (qp):
    """H is the Hamiltonian for the system."""

    return K(qp) + U(qp)

def H__conservation_test ():
    """
    This test verifies that H is conserved along the flow of H (just a sanity check, this fact
    is easily provable in general).
    """

    qp = phase_space_coordinates()
    #X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp)
    H_qp = H__(*qp.reshape(-1).tolist())
    X_H = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    # Sanity check
    X_H__H = vorpy.manifold.apply_vector_field_to_function(X_H, H_qp, qp)
    if X_H__H != 0:
        raise FancyException(f'Expected X_H(H) == 0 but instead got {X_H__H}')
    print('H__conservation_test passed')

def p_theta (qp):
    """p_theta is the angular momentum for the system and is conserved along solutions."""

    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    return x*p_y - y*p_x

def p_theta__conservation_test ():
    """
    This test verifies that p_theta is conserved along the flow of H.
    """

    qp = phase_space_coordinates()
    X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp)

    # Sanity check
    X_H__p_theta = vorpy.manifold.apply_vector_field_to_function(X_H, p_theta(qp), qp)
    if X_H__p_theta != 0:
        raise FancyException(f'Expected X_H(p_theta) == 0 but instead got {X_H__p_theta}')
    print('p_theta__conservation_test passed')

def J (X):
    """J can be thought of as "dilational momentum" for the system, and is conserved along solutions when H = 0."""

    x,y,z       = X[0,:]
    p_x,p_y,p_z = X[1,:]

    return x*p_x + y*p_y + 2*z*p_z

def J__restricted_conservation_test ():
    """This test verifies that J is conserved along the flow of H if restricted to the H = 0 submanifold."""

    qp = phase_space_coordinates()
    H_qp = H(qp)
    X_H = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    J_qp = J(qp)
    X_H__J = vorpy.manifold.apply_vector_field_to_function(X_H, J_qp, qp)

    p_z = qp[1,2]

    # Solve for p_z in H_qp == 0; there are two sheets to this solution.
    p_z_solution_v = sp.solve(H_qp, p_z)
    assert len(p_z_solution_v) == 2, f'Expected 2 solutions for p_z in H == 0, but instead got {len(p_z_solution_v)}'
    #print('There are {0} solutions for the equation: {1} = 0'.format(len(p_z_solution_v), H_qp))
    #for i,p_z_solution in enumerate(p_z_solution_v):
        #print('    solution {0}: p_z = {1}'.format(i, p_z_solution))

    for solution_index,p_z_solution in enumerate(p_z_solution_v):
        # We have to copy X_H__J or it will only be a view into X_H__J and will modify the original.
        # The [tuple()] access is to obtain the scalar value out of the
        # sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray object that doit() returns.
        X_H__J__restricted = sp.Subs(np.copy(X_H__J), p_z, p_z_solution).doit()[tuple()].simplify()
        #print(f'solution_index = {solution_index}, X_H__J__restricted = {X_H__J__restricted}')
        if X_H__J__restricted != 0:
            raise FancyException(f'Expected X_H__J__restricted == 0 for solution_index = {solution_index}, but actual value was {X_H__J__restricted}')

    print('J__restricted_conservation_test passed')

def J__test ():
    """This test verifies that dJ/dt = 2*H."""

    qp              = phase_space_coordinates()
    qp_             = qp.reshape(-1).tolist()

    x,y,z           = qp[0,:]
    p_x,p_y,p_z     = qp[1,:]

    P_x_            = P_x__(*qp_)
    P_y_            = P_y__(*qp_)
    mu_             = mu__(x,y,z)
    r_squared_      = r_squared__(x,y)

    H_qp            = H__(*qp_)
    X_H             = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    J_qp            = J__(*qp_)
    # Because X_H gives the vector field defining the time derivative of a solution to the dynamics,
    # it follows that X_H applied to J is equal to dJ/dt (where J(t) is J(qp(t)), where qp(t) is a
    # solution to Hamilton's equations).
    X_H__J          = vorpy.manifold.apply_vector_field_to_function(X_H, J_qp, qp)
    #print(f'J__test; X_H__J = {X_H__J}')
    #print(f'J__test; 2*H = {sp.expand(2*H_qp)}')
    actual_value    = X_H__J - sp.expand(2*H_qp)
    #print(f'J__test; X_H__J - 2*H = {actual_value}')

    # Annoyingly, this doesn't simplify to 0 automatically, so some manual manipulation has to be done.

    # Manipulate the expression to ensure the P_x and P_y terms cancel
    actual_value    = sp.collect(actual_value, [P_x_, P_y_])
    #print(f'J__test; after collect P_x, P_y: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [P_x_, P_y_], [P_x_._expanded(), P_y_._expanded()]).doit()
    #print(f'J__test; after subs P_x, P_y: X_H__J - 2*H = {actual_value}')

    # Manipulate the expression to ensure the mu terms cancel
    actual_value    = sp.factor_terms(actual_value, clear=True, fraction=True)
    #print(f'J__test; after factor_terms: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.collect(actual_value, [r_squared_])
    #print(f'J__test; after collect r_squared_: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [r_squared_._expanded()], [r_squared_]).doit()
    #print(f'J__test; after subs r_squared: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [mu_._expanded()], [mu_]).doit()
    #print(f'J__test; after subs mu: X_H__J - 2*H = {actual_value}')

    if actual_value != 0:
        raise FancyException(f'Expected X_H__J - 2*H == 0, but actual value was {actual_value}')

    print('J__test passed')

def A (qp):
    """
    A is the standard contact form in R^3, taken as a differential form in T*(R^3), then V is its symplectic dual.

    A = dz + y/2 * dx - x/2 * dy
    """
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]
    return np.array((
        (          y/2,          -x/2, sp.Integer(1)),
        (sp.Integer(0), sp.Integer(0), sp.Integer(0)),
    ))

def V (qp):
    """
    If A is the standard contact form in R^3, taken as a differential form in T*(R^3), then V is its symplectic dual.

    A = dz + y/2 * dx - x/2 * dy
    V = del_{p_z} + y/2 * del_{p_x} - x/2 * del_{p_y}
    """
    return vorpy.symplectic.symplectic_dual_of_covector_field(A(qp))

def V__test ():
    qp = phase_space_coordinates()

    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    expected_value = np.array((
        (sp.Integer(0), sp.Integer(0), sp.Integer(0)),
        (         -y/2,           x/2,            -1),
    ))
    actual_value = V(qp)
    error = actual_value - expected_value
    if not np.all(error == 0):
        raise FancyException(f'Expected V = {expected_value} but it was actually {actual_value}')
    print('V__test passed')

def lie_bracket_of__X_H__V__test ():
    qp = phase_space_coordinates()
    #print(f'H = {H(qp)}')
    #print(f'X_H = {vorpy.symplectic.symplectic_gradient_of(H(qp), qp)}')

    #print(f'A = {A(qp)}')
    #print(f'V = {V(qp)}')

    lb__X_H__V = vorpy.manifold.lie_bracket(vorpy.symplectic.symplectic_gradient_of(H(qp), qp), V(qp), qp)
    #print(f'[X_H,V] = {lb__X_H__V}')

    # NOTE: This has sign opposite from Corey's Weinstein Note PDF file (he has a sign
    # error in computing the symplectic dual of V).
    expected__lb__X_H__V = np.array((
        (sp.Integer(0), sp.Integer(0), sp.Integer(0)),
        (     -P_y(qp),       P_x(qp), sp.Integer(0)),
    ))
    #print(f'expected value = {expected__lb__X_H__V}')
    #print(f'[X_H,V] - expected_value = {lb__X_H__V - expected__lb__X_H__V}')

    if not np.all(lb__X_H__V == expected__lb__X_H__V):
        raise FancyException(f'Expected [X_H,V] = {expected__lb__X_H__V} but it was actually {lb__X_H__V}')

    print('lie_bracket_of__X_H__V__test passed')

"""
Design notes for integral-of-motion search.

Assume F(qp) is a polynomial of degree n for a given n and that F is an integral of motion.
F being an integral of motion is defined by it being conserved along solutions to the system,
and therefore is equivalent to X_H(F) = 0.  Thus, represent F as a generic symbolic polynomial,
and then attempt to solve for its coefficients within X_H(F) = 0.  Because angular momentum
is a conserved polynomial (in this case of degree 2), this method should find angular momentum).

Additionally, if we restrict the search to the H = 0 manifold, this condition can be applied
after evaluating X_H(F) and before attempting to solve for the coefficients of F.  Because J
is conserved in this case (only when H = 0), this method should find J.
"""

#def tensor_power (V, p):
    #"""
    #Returns the pth tensor power of vector V.  This should be a tensor having order p,
    #which looks like V \otimes ... \otimes V (with p factors).  If p is zero, then this
    #returns 1.

    #TODO: Implement this for tensors of arbitrary order (especially including 0-tensors).
    #"""

    #V_order = vorpy.tensor.order(V)
    #if V_order != 1:
        #raise FancyException(f'Expected V to be a vector (i.e. a 1-tensor), but it was actually a {V_order}-tensor')
    #if p < 0:
        #raise FancyException(f'Expected p to be a nonnegative integer, but it was actually {p}')

    #if p == 0:
        #return np.array(1) # TODO: Should this be an actual scalar?
    #elif p == 1:
        #return V
    #else:
        #assert len(V.shape) == 1 # This should be equivalent to V_order == 1.
        #V_dim = V.shape[0]
        #V_to_the_p_minus_1 = vorpy.tensor.tensor_power_of_vector(V, p-1)
        #retval_shape = (V_dim,)*p
        #return np.outer(V, V_to_the_p_minus_1.reshape(-1)).reshape(*retval_shape)

def tensor_power__test ():
    V = np.array((sp.var('x'), sp.var('y'), sp.var('z')))

    #print(f'V = {V}')
    #for p in range(5):
        #print(f'vorpy.tensor.tensor_power_of_vector(V, {p}):')
        #print(f'{vorpy.tensor.tensor_power_of_vector(V, p)}')
        #print()

    # Specific comparisons

    power           = 0
    expected_value  = 1
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 1
    expected_value  = V
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 2
    expected_value  = vorpy.tensor.contract('i,j', V, V, dtype=object)
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 3
    expected_value  = vorpy.tensor.contract('i,j,k', V, V, V, dtype=object)
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 4
    expected_value  = vorpy.tensor.contract('i,j,k,l', V, V, V, V, dtype=object)
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    print('tensor_power__test passed')

# TODO: Write symbolic_homogeneous_polynomial function and use it to implement symbolic_polynomial function.

def symbolic_polynomial (coefficient_prefix, degree, X):
    """
    Returns a generic polynomial of the given degree with symbolic coefficients,
    as well as a list of the coefficients.  X is the coordinates to express the
    polynomial in.  Each polynomial term does not include multiplicity (e.g.
    the `x*y` term would appear as `a_0_1*x*y`, not as `2*a_0_1*x*y`).

    The return value is polynomial, coefficients.
    """
    # TODO: Allow specification of which degrees should be present in this polynomial

    X_reshaped = X.reshape(-1)

    coefficient_accumulator = []
    polynomial_accumulator = sp.Integer(0)
    # TEMP HACK: Omitting the 0-degree term for now.
    for p in range(1, degree+1):
        degree_shape                = (X_reshaped.size,)*p
        degree_p_coefficients       = vorpy.symbolic.tensor(coefficient_prefix, degree_shape)
        # TODO: Have to encode the symmetries in the coefficients -- in particular, could replace any
        # coefficient with non-strictly-increasing indices with the corresponding one that has
        # strictly increasing indices.
        for I in vorpy.tensor.multiindex_iterator(degree_shape):
            # Replace the non-strictly-increasing-indexed coefficients with 0, and store the rest for return.
            if I != tuple(sorted(I)):
                degree_p_coefficients[I] = 0
            else:
                coefficient_accumulator.append(degree_p_coefficients[I])

        degree_p_variable_tensor    = vorpy.tensor.tensor_power_of_vector(X_reshaped, p)
        # Because of the sparsification done above, multiplying it out this way is somewhat inefficient, but it's fine for now.
        polynomial_accumulator     += np.dot(degree_p_coefficients.reshape(-1), degree_p_variable_tensor.reshape(-1))

    return polynomial_accumulator, coefficient_accumulator

def symbolic_polynomial__test ():
    X = np.array((sp.var('x'), sp.var('y'), sp.var('z')))

    print(f'vorpy.symbolic.symbolic_polynomial("a", 2, {X}) = {vorpy.symbolic.symbolic_polynomial("a", 2, X)}')
    # TODO: actually do a check.

def collect_by_linear_factors (expr, linear_factor_v, *, term_procedure=None, sanity_check=False):
    #expr_order = vorpy.tensor.order(expr)
    #if expr_order != 0:
        #raise FancyException(f'Expected expr to have tensor order 0, but it was actually {expr_order}')
    print(f'collect_by_linear_factors; expr = {expr}, linear_factor_v = {linear_factor_v}')
    dexpr = vorpy.symbolic.differential(expr, linear_factor_v)
    print(f'collect_by_linear_factors; dexpr = {dexpr}')

    if term_procedure is not None:
        dexpr_reshaped = dexpr.reshape(-1) # This is just a view into dexpr.
        for i in range(dexpr_reshaped.size):
            dexpr_reshaped[i] = term_procedure(dexpr_reshaped[i])
        print(f'collect_by_linear_factors; after term_procedure: dexpr = {dexpr}')

    if sanity_check:
        ddexpr = vorpy.symbolic.differential(dexpr, linear_factor_v)
        if not np.all(ddexpr == 0):
            raise FancyException(f'Factors did not occur linearly in expr; expected Hessian of expr to be 0.')

    return np.dot(dexpr, linear_factor_v)

def find_integral_of_motion (highest_degree_polynomial):
    """
    TODO: Could potentially use

        https://www.sciencedirect.com/science/article/pii/S0747717185800146
        https://www.sciencedirect.com/science/article/pii/S0747717185800146/pdf?md5=2523d9cdea9c529ac03075da71605760&pid=1-s2.0-S0747717185800146-main.pdf

    to deal with a larger class of functions (ones involving radicals).
    """

    qp = phase_space_coordinates()
    H_qp = H(qp)
    X_H = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    qp_reshaped = qp.reshape(-1)

    for degree in range(1, highest_degree_polynomial+1):
        print(f'degree = {degree}')
        F, F_coefficients = vorpy.symbolic.symbolic_polynomial('F', degree, qp)
        print(f'F = {F}')
        print(f'len(F_coefficients) = {len(F_coefficients)}')
        print(f'F_coefficients = {F_coefficients}')
        X_H__F = vorpy.manifold.apply_vector_field_to_function(X_H, F, qp)
        print(f'\nOriginal expression')
        print(f'X_H__F = {X_H__F}')
        X_H__F = sp.fraction(sp.factor_terms(X_H__F, clear=True))[0]
        print(f'\nAfter taking top of fraction:')
        print(f'X_H__F = {X_H__F}')

        X_H__F = collect_by_linear_factors(X_H__F, np.array(F_coefficients), term_procedure=lambda e:sp.simplify(e), sanity_check=True)
        print(f'\nAfter collect_by_linear_factors:')
        print(f'X_H__F = {X_H__F}')

        #X_H__F = sp.expand_mul(X_H__F)
        #print(f'\nAfter expand_mul:')
        #print(f'X_H__F = {X_H__F}')
        #X_H__F = sp.collect(X_H__F, F_coefficients)
        #print(f'\nAfter collecting by coefficients:')
        #print(f'X_H__F = {X_H__F}')
        #X_H__F = sp.simplify(X_H__F)
        #print(f'\nAfter simplifying:')
        #print(f'X_H__F = {X_H__F}')

        # Look for integrals of motion restricted to H = 0 submanifold.
        if True:
            p_z = qp[1,2]
            # Solve for p_z in H_qp == 0; there are two sheets to this solution.
            p_z_solution_v = sp.solve(H_qp, p_z)
            assert len(p_z_solution_v) == 2, f'Expected 2 solutions for p_z in H == 0, but instead got {len(p_z_solution_v)}'
            #print('There are {0} solutions for the equation: {1} = 0'.format(len(p_z_solution_v), H_qp))
            #for i,p_z_solution in enumerate(p_z_solution_v):
                #print('    solution {0}: p_z = {1}'.format(i, p_z_solution))

            #for solution_index,p_z_solution in enumerate(p_z_solution_v):
            if True:
                # TEMP HACK: Just use sheet 0 of the solution for now.
                solution_index = 0
                p_z_solution = p_z_solution_v[0]
                X_H__F = sp.Subs(X_H__F, p_z, p_z_solution).doit().simplify()

                print(f'X_H__F restricted to H = 0: {X_H__F}')

        coefficient_to_solve_for_v = copy.deepcopy(F_coefficients)
        total_substitution_v = []

        # TODO: using multiindex_iterator is just a cheap way to get nice test tuples
        # of the form (0,0,0,0,0,0), (0,0,0,0,0,1), etc.  Should really define the range
        # of each test component and use itertools.product.  This plugging in of specific
        # values and then solving for coefficients is a dumb but effective way of avoiding
        # having to determine a linearly independent set of nonlinear functions of the
        # qp variables.
        #
        # TODO: Ideally, we would use test values that are not invariant under the symmetries
        # of the problem.  Though determining this is relatively hard.
        for qp_substitutions in vorpy.tensor.multiindex_iterator((2,)*qp_reshaped.size):
            qp_substitutions = np.array(qp_substitutions)
            # Skip (x,y,z) == 0.
            if np.all(qp_substitutions[0:3] == 0):
                print(f'Skipping qp_substitutions = {qp_substitutions} to avoid division by zero')
                continue

            # We have to copy X_H__F or it will only be a view into X_H__F and will modify the original.
            # The [tuple()] access is to obtain the scalar value out of the
            # sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray object that doit() returns.
            #particular_expression = sp.Subs(np.copy(X_H__F), qp_reshaped, qp_substitutions).doit()[tuple()].simplify()
            particular_expression = sp.collect(sp.Subs(np.copy(X_H__F), qp_reshaped, qp_substitutions).doit()[tuple()].simplify(), coefficient_to_solve_for_v)

            print(f'len(coefficient_to_solve_for_v) = {len(coefficient_to_solve_for_v)}')
            print(f'coefficient_to_solve_for_v = {coefficient_to_solve_for_v}')
            print(f'running solver on expression: {particular_expression}')
            solutions = sp.solve(particular_expression, coefficient_to_solve_for_v)
            print(f'qp_substitutions = {qp_substitutions}')
            print(f'solutions = {solutions}')
            # TODO: Any time a particular solution is found for any coefficient, replace that coefficient
            # with that solution and record that replacement.  This will narrow down the search field.
            # If there are multiple solutions, this necessarily forces the solution search to branch,
            # which will require a different search function design.

            # TEMP HACK: For now, assume there will be no branching, so just use the first solution, if any.
            if len(solutions) > 0:
                if len(solutions) > 1:
                    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print(f'WARNING: There was more than one solution -- branching search function is needed.')
                    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                substitution_var_v = []
                substitution_value_v = []

                first_solution = solutions[0]
                # Make sure that we can call zip on first_solution with coefficient_to_solve_for_v
                if not isinstance(first_solution, tuple):
                    first_solution = (first_solution,)

                # Substitute the first solution
                for var,value in zip(coefficient_to_solve_for_v, first_solution):
                    if var != 0 and var != value and value != sp.nan:
                        total_substitution_v.append([var, value])
                        substitution_var_v.append(var)
                        substitution_value_v.append(value)

                assert len(substitution_var_v) == len(substitution_value_v)
                there_was_a_substitution = len(substitution_var_v) > 0
                if there_was_a_substitution:
                    print(f'Substituting:')
                    for var, value in zip(substitution_var_v, substitution_value_v):
                        print(f'    {var} |-> {value}')

                    # Substitute the solutions into the conservation law expression
                    #X_H__F = sp.Subs(X_H__F, substitution_var_v, substitution_value_v).doit().simplify()
                    X_H__F = sp.collect(sp.Subs(X_H__F, substitution_var_v, substitution_value_v).doit().simplify(), coefficient_to_solve_for_v)
                    # Substitute the solutions into each of the existing solution substitutions
                    # so that the substitution expressions don't depend on each other and will
                    # boil down to constants once all coefficients have been solved for.
                    for i in range(len(total_substitution_v)):
                        total_substitution_v[i][1] = sp.Subs(total_substitution_v[i][1], substitution_var_v, substitution_value_v).doit().simplify()
                    print(f'After substitutions, X_H__F = {X_H__F}')
                    print()
                for substitution_var in substitution_var_v:
                    coefficient_to_solve_for_v.remove(substitution_var)

            print(f'total_substitution_v:')
            for var,value in total_substitution_v:
                print(f'    {var} |-> {value}')
            print(f'{len(coefficient_to_solve_for_v)} coefficients still to solve for: {coefficient_to_solve_for_v}')

            if X_H__F == 0:
                break

            print()


        #solutions = sp.solve(X_H__F, F_coefficients)
        #print(f'solutions = {solutions}')

        print()

if __name__ == '__main__':
    P_x__test()
    if True:
        J__test()
    if True:
        lie_bracket__test()
        lie_bracket_of__X_H__V__test()
        V__test()
    if True:
        H__conservation_test()
    if True:
        p_theta__conservation_test()
        J__restricted_conservation_test()
        tensor_power__test()
        symbolic_polynomial__test()
    if False:
        find_integral_of_motion(2)
