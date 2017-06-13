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

    class PendulumNd:
        """
        Defines the various geometric-mechanical structures of a spherical pendulum in arbitrary dimension.

        Coordinates are assumed to have shape (2,N), i.e. np.array([q,p]), where q and p are the vector-valued
        pendular angle and momentum respectively.  The angle coordinates are assumed to be normal coordinates
        with origin denoting the downward, stable equilibrium position.  The potential energy function is the
        vertical position of the pendular mass.
        """

        @staticmethod
        def K (p):
            return 0.5*np.sum(np.square(p))

        @staticmethod
        def V (q):
            # np.linalg.norm(q) gives the angle from the vertical axis
            return np.cos(np.linalg.norm(q))

        @staticmethod
        def H (coordinates):
            q = coordinates[0,:]
            p = coordinates[1,:]
            return PendulumNd.K(p) - PendulumNd.V(q)

        @staticmethod
        def dK_dp (p):
            return p

        @staticmethod
        def dV_dq (q):
            norm_q = np.linalg.norm(q)
            return np.sin(norm_q)/norm_q * q

        @staticmethod
        def X_H (coordinates, *args): # args is assumed to be the time coordinate and other ignored args
            q = coordinates[0,:]
            p = coordinates[1,:]
            # This is the symplectic gradient of H.
            return np.array((PendulumNd.dK_dp(p), -PendulumNd.dV_dq(q)))

    class Results:
        def __init__ (self):
            self.result_name_v = []
            self.result_d = {}

        def add_result (self, result_name, t_v, qp_v):
            self.result_name_v.append(result_name)
            self.result_d[result_name] = {
                'N'                     :qp_v.shape[-1],
                'can_plot_phase_space'  :N == 1,
                't_v'                   :t_v,
                'qp_v'                  :qp_v,
                'H_v'                   :vorpy.apply_along_axes(PendulumNd.H, (-2,-1), qp_v, output_axis_v=(), func_output_shape=()),
            }
            assert len(self.result_name_v) == len(self.result_d)

        def plot_result (self, result_name, axis_v):
            result                  = self.result_d[result_name]
            N                       = result['N']
            can_plot_phase_space    = result['can_plot_phase_space']
            t_v                     = result['t_v']
            qp_v                    = result['qp_v']
            H_v                     = result['H_v']

            can_plot_phase_space = N == 1
            T = qp_v.shape[0]
            assert H_v.shape[0] == T

            H_v_reshaped = H_v.reshape(T, -1)
            # Detrend each H_v, so that the plot is just the deviation from the mean and the min and max are more meaningful
            H_v_reshaped -= np.mean(H_v_reshaped, axis=0)

            min_H   = np.min(H_v_reshaped)
            max_H   = np.max(H_v_reshaped)
            range_H = max_H - min_H

            axis    = axis_v[0]
            axis.set_title('Hamiltonian (detrended); range = {0:.2e}\nmethod: {1}'.format(range_H, result_name))
            axis.plot(t_v, H_v_reshaped)
            axis.axhline(min_H, color='green')
            axis.axhline(max_H, color='green')

            sqd_v   = vorpy.apply_along_axes(lambda x:np.sum(np.square(x)), (-2,-1), qp_v - qp_v[0])
            sqd_v_reshaped = sqd_v.reshape(T, -1)

            axis    = axis_v[1]
            axis.set_title('{0} : sq dist from initial'.format(result_name))
            axis.semilogy(t_v, sqd_v_reshaped)

            # We can only directly plot the phase space if the configuration space is 1-dimensional.
            if can_plot_phase_space:
                # This is to get rid of the last axis, which has size 1, and therefore can be canonically reshaped away
                # via the canonical identification between 1-vectors and scalars.
                qp_v_reshaped = qp_v.reshape(T, -1, 2)
                print('qp_v.shape = {0}'.format(qp_v.shape))
                print('qp_v_reshaped.shape = {0}'.format(qp_v_reshaped.shape))

                axis = axis_v[2]
                axis.set_title('phase space\nmethod: {0}'.format(result_name))
                axis.set_aspect('equal')
                axis.plot(qp_v_reshaped[...,0], qp_v_reshaped[...,1])

        def plot (self, filename):
            any_can_plot_phase_space = any(result['can_plot_phase_space'] for _,result in self.result_d.items())

            row_count = len(self.result_name_v)
            col_count = 3 if any_can_plot_phase_space else 2
            fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(8*col_count, 8*row_count))

            for result_name,axis_v in zip(self.result_name_v,axis_vv):
                self.plot_result(result_name, axis_v)

            fig.tight_layout()
            plt.savefig(filename)
            print('wrote to file "{0}"'.format(filename))

    def compare_integrator_methods (N):
        results = Results()

        t_v = np.linspace(0.0, 30.0, 1500)
        qp_0 = np.zeros((2,N), dtype=float)
        qp_0[0,0] = np.pi/2.0

        # scipy.integrate.odeint expects the phase space coordinate to have shape (2*N,), not (2,N).
        # This function adapts between those two conventions.
        def X_H_adapted_for_odeint (coordinates_as_1_tensor, *args):
            qp = coordinates_as_1_tensor.reshape(2,-1)      # -1 will cause reshape to infer the value of N.
            return PendulumNd.X_H(qp, *args).reshape(-1)    # -1 will cause reshape to produce a 1-tensor.

        results.add_result('standard odeint', t_v, scipy.integrate.odeint(X_H_adapted_for_odeint, qp_0.reshape(-1), t_v).reshape(len(t_v),2,-1))
        results.add_result('symplectic Euler', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.euler))
        results.add_result('symplectic Verlet', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.verlet))
        results.add_result('symplectic Ruth3', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.ruth3))
        results.add_result('symplectic Ruth4', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.ruth4))

        filename = 'symplectic_integration.Pendulum{0}d.png'.format(N)
        results.plot(filename)

    def test_parallel_initial_conditions (N, shape_of_parallelism):
        rng = np.random.RandomState(42)

        t_v = np.linspace(0.0, 30.0, 1500)
        base_shape = (2,N)
        base_qp_0 = np.zeros(base_shape, dtype=float)
        base_qp_0[0,0] = np.pi/2.0

        qp_0 = rng.randn(*(shape_of_parallelism+base_shape)) + base_qp_0

        results = Results()

        results.add_result('symplectic Euler', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.euler))
        results.add_result('symplectic Verlet', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.verlet))
        results.add_result('symplectic Ruth3', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.ruth3))
        results.add_result('symplectic Ruth4', t_v, symplectic_integrate_split_hamiltonian(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=predefined_method_coefficients.ruth4))

        filename = 'symplectic_integration_parallel.Pendulum{0}d.shape:{1}.png'.format(N, shape_of_parallelism)
        results.plot(filename)

    for N in [1,2,3]:
        compare_integrator_methods(N)

    for N,shape_of_parallelism in itertools.product([1,2], [(),(2,),(2,3)]):
        test_parallel_initial_conditions(N, shape_of_parallelism)

