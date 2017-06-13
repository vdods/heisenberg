import collections
import numpy as np
import vorpy

"""
Implements some symplectic integrators.  This is a class of integrators which
preserve the symplectic form on the cotangent bundle of phase space.  Coordinates
on phase space are typically written as (q,p), which denote the position and
momentum coordinates respectively.  A symplectic integrator will then integrate
Hamilton's equations

    dq/dt =   \partial H / \partial p
    dp/dt = - \partial H / \partial q

where H(q,p) is the Hamiltonian (aka total energy) of the system.

In particular, this module implements predefined_method_coefficients to integrate systems with "split"
Hamiltonians.  A split Hamiltonian has the form

    H(q,p) = K(p) + V(q)

where K and V are the kinetic and potential energy functions, respectively.

References

    https://en.wikipedia.org/wiki/Symplectic_integrator
    https://en.wikipedia.org/wiki/Energy_drift
"""

def __make_ruth4_method_coefficients ():
    cbrt_2 = 2.0**(1.0/3.0)
    b = 2.0 - cbrt_2
    c_0 = c_3 = 0.5/b
    c_1 = c_2 = 0.5*(1.0 - cbrt_2)/b
    d_0 = d_2 = 1.0/b
    d_1 = -cbrt_2/b
    d_3 = 0.0
    return np.array([
        [c_0, c_1, c_2, c_3],
        [d_0, d_1, d_2, d_3]
    ])

PredefinedMethodCoefficientsType = collections.namedtuple('PredefinedMethodCoefficientsType', ['euler', 'verlet', 'ruth3', 'ruth4'])
predefined_method_coefficients = PredefinedMethodCoefficientsType(
    # euler
    np.array([
        [1.0],
        [1.0]
    ]),
    # verlet
    np.array([
        [0.0, 1.0],
        [0.5, 0.5]
    ]),
    # ruth3
    np.array([
        [1.0, -2.0/3.0, 2.0/3.0],
        [-1.0/24.0, 0.75, 7.0/24.0]
    ]),
    # ruth4
    __make_ruth4_method_coefficients()
)

def symplectic_integrate_split_hamiltonian (*, initial_coordinates, t_v, dK_dp, dV_dq, coefficients):
    """
    This function computes multiple timesteps of the symplectic integrator defined by the method
    parameter

    Let N denote the dimension of the configuration space (i.e. the number of components of the q coordinate).

    A single set of coordinates shall be represented with a numpy array of shape (2,N).

    Parameters:

    -   initial_coordinates specify the coordinates from which to begin integrating.  This should have
        the shape (A_1,A_2,...,A_M,2,N), where M might be zero (in which case the shape is (2,N)).
        The indices A_1,A_2,...,A_M (of which there can be none) may index some other parameter to
        the initial conditions, such that many integral curves will be computed in parallel (one for
        each assignment of A_1,A_2,...,A_M index).

    -   t_v specifies a list of the time values at which to integrate the system.  The first value corresponds
        to the initial condition, so the length of t_v must be at least 1.  The timesteps are computed as the
        difference between successive elements.  The timesteps can be negative; see
        https://en.wikipedia.org/wiki/Symplectic_integrator#A_second-order_example

    -   dK_dp and dV_dq should be functions of the respective forms

        lambda p : <expression evaluating \partial K / \partial p>
        lambad q : <expression evaluating \partial V / \partial q>

        and should each accept and return a vector having N components.

    -   coefficients should be a numpy.ndarray with shape (2,K), where K is the order of the integrator.
        These coefficients define the specific integrator by defining the weight of each leapfrog update
        step.  Row 0 and row 1 correspond to the update step weight for even and odd leapfrog update steps
        respectively.  Predefined coefficients are available via the predefined_method_coefficients variable.
        In particular,

            predefined_method_coefficients.euler  : 1st order
            predefined_method_coefficients.verlet : 2nd order
            predefined_method_coefficients.ruth3  : 3rd order
            predefined_method_coefficients.ruth4  : 4rd order

        The coefficients' rows must sum to one (i.e. numpy.sum(coefficients[i]) == 1.0 for i in [0,1]), and are
        described at https://en.wikipedia.org/wiki/Symplectic_integrator

    Return values:

    -   integrated_coordinates is a numpy.ndarray having shape (len(t_v),A_1,A_2,...,A_M,2,N), containing the coordinates of
        each integrator step starting with initial_coordinates.
    """

    initial_coordinates_shape = np.shape(initial_coordinates)
    coefficients_shape = np.shape(coefficients)

    assert len(initial_coordinates_shape) >= 2
    assert initial_coordinates_shape[-2] == 2
    assert len(t_v) >= 1
    assert coefficients_shape[0] == 2, 'coefficients must have shape (2,K), where K is the order of the integrator.'
    assert np.allclose(np.sum(coefficients, axis=1), 1.0), 'rows of coefficients matrix must sum to 1.0 (within numerical tolerance)'

    # N is the dimension of the underlying configuration space.  Thus 2*N is the dimension of the phase space,
    # hence a coordinate of the phase space having shape (2,N).
    N = initial_coordinates_shape[-1]
    # get the axes not corresponding to the final (2,N) part of the shape.  This can be the empty tuple.
    non_coordinate_shape = initial_coordinates_shape[:-2]
    non_coordinate_axis_v = tuple(range(len(non_coordinate_shape)))
    # T is the number of timesteps
    T = len(t_v)
    # order is the order of the integrator (number of coefficients in each row).
    order = coefficients_shape[1]

    # Create the return value
    integrated_coordinates = np.ndarray((T,)+non_coordinate_shape+(2,N), dtype=initial_coordinates.dtype)
    # Create a buffer for intermediate coordinates
    current_coordinates = np.copy(initial_coordinates)

    # Create slices to address the q and p components of current_coordinates.
    q = current_coordinates[...,0,:]
    p = current_coordinates[...,1,:]

    for step_index,timestep in enumerate(np.diff(t_v)):
        integrated_coordinates[step_index,...] = current_coordinates
        # Iterate over (c,d) pairs and perform the leapfrog update steps.
        for c,d in zip(coefficients[0],coefficients[1]):
            # The (2,N) phase space is indexed by the last two indices, i.e. (-2,-1) in that order.
            q += timestep*c*vorpy.apply_along_axes(dK_dp, (-1,), p, output_axis_v=(-1,), func_output_shape=(N,))
            p -= timestep*d*vorpy.apply_along_axes(dV_dq, (-1,), q, output_axis_v=(-1,), func_output_shape=(N,))

    integrated_coordinates[T-1,...] = current_coordinates

    return integrated_coordinates

if __name__ == '__main__':
    import itertools
    import matplotlib.pyplot as plt
    import scipy.integrate

    def test_stuff ():
        class Pendulum:
            N = 1

            @staticmethod
            def H (coordinates):
                """coordinates is np.array([[q],[p]]), where q and p are the pendular angle and momentum respectively."""
                return 0.5*coordinates[1,0]**2 - np.cos(coordinates[0,0])

            @staticmethod
            def dK_dp (p):
                return p

            @staticmethod
            def dV_dq (q):
                return np.sin(q)

            @staticmethod
            def X_H (coordinates, *args): # args is assumed to be the time coordinate and other ignored args
                return np.array([coordinates[1], -np.sin(coordinates[0])])

            @staticmethod
            def apply_H (coordinates):
                assert coordinates.shape[-1] == Pendulum.N
                assert coordinates.shape[-2] == 2
                retval = np.ndarray(coordinates.shape[:-2], dtype=coordinates.dtype)
                for I in itertools.product(*[range(s) for s in retval.shape]):
                    retval[I] = Pendulum.H(coordinates[I])
                return retval

        result_name_v = []
        result_d = {}

        def add_result (result_name, t_v, qp_v):
            nonlocal result_name_v
            nonlocal result_d

            result_name_v.append(result_name)
            result_d[result_name] = {
                't_v':t_v,
                'qp_v':qp_v,
                'H_v':Pendulum.apply_H(qp_v),
            }

        def plot_result (result_name, axis_v):
            nonlocal result_d

            result = result_d[result_name]
            qp_v = result['qp_v']
            H_v = result['H_v']

            T = qp_v.shape[0]

            # This is to get rid of the last axis, which has size 1, and therefore can be canonically reshaped away
            # via the canonical identification between 1-vectors and scalars.
            qp_v_reshaped = qp_v.reshape(T, 2)

            axis = axis_v[0]
            axis.set_title('phase space\nmethod: {0}'.format(result_name))
            axis.set_aspect('equal')
            axis.plot(qp_v_reshaped[:,0], qp_v_reshaped[:,1])

            min_H = np.min(H_v)
            max_H = np.max(H_v)
            range_H = max_H - min_H

            axis = axis_v[1]
            axis.set_title('Hamiltonian; range = {0:.2e}\nmethod: {1}'.format(range_H, result_name))
            axis.plot(t_v, H_v)
            #axis.plot(H_v)
            axis.axhline(min_H, color='green')
            axis.axhline(max_H, color='green')

            # sqd_v = np.sum(np.square(qp_v_reshaped - qp_v_reshaped[0]), axis=-1)

            # axis = axis_v[2]
            # axis.set_title('{0} : sq dist from initial'.format(result_name))
            # axis.semilogy(t_v, sqd_v)

        # t_v = np.linspace(0.0, 10.0, 5000)
        t_v = np.linspace(0.0, 30.0, 1500)
        qp_0 = np.array([[np.pi/2.0], [0.0]])

        add_result('standard odeint', t_v, scipy.integrate.odeint(Pendulum.X_H, qp_0.reshape((2,)), t_v).reshape(len(t_v),2,Pendulum.N))
        add_result('symplectic Euler', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, coefficients=predefined_method_coefficients.euler))
        add_result('symplectic Verlet', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, coefficients=predefined_method_coefficients.verlet))
        add_result('symplectic Ruth3', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, coefficients=predefined_method_coefficients.ruth3))
        add_result('symplectic Ruth4', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, coefficients=predefined_method_coefficients.ruth4))

        assert len(result_name_v) == len(result_d)

        row_count = len(result_name_v)
        col_count = 2
        fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(8*col_count, 8*row_count))

        for result_name,axis_v in zip(result_name_v,axis_vv):
            plot_result(result_name, axis_v)

        fig.tight_layout()
        filename = 'symplectic_integration.png'
        plt.savefig(filename)
        print('wrote to file "{0}"'.format(filename))

    test_stuff()
