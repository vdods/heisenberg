import numpy as np

"""
Implements some symplectic integrators.  This is a class of integrators which
preserve the symplectic form on the cotangent bundle of phase space.  Coordinates
on phase space are typically written as (q,p), which denote the position and
momentum coordinates respectively.  A symplectic integrator will then integrate
Hamilton's equations

    dq/dt =   \partial H / \partial p
    dp/dt = - \partial H / \partial q

where H(q,p) is the Hamiltonian (aka total energy) of the system.

In particular, this module implements methods to integrate systems with "split"
Hamiltonians.  A split Hamiltonian has the form

    H(q,p) = K(p) + V(q)

where K and V are the kinetic and potential energy functions, respectively.

References

    https://en.wikipedia.org/wiki/Symplectic_integrator
    https://en.wikipedia.org/wiki/Energy_drift
"""

def symplectic_integrator_split_hamiltonian (*, initial_coordinates, dK_dp, dV_dq, t_v, c_v, d_v, return_intermediates=False):
    """
    This function computes a single timestep of the symplectic integrator defined by (c_v, d_v).

    Let N denote the dimension of the configuration space (i.e. the number of components of the q coordinate).

    coordinates shall be represented with a numpy array of shape (2,N).

    Parameters:

    -   initial_coordinates specify the coordinates from which to begin integrating.

    -   dK_dp and dV_dq should be functions of the respective forms

        lambda p : <expression evaluating \partial K / \partial p>
        lambad q : <expression evaluating \partial V / \partial q>

        and should each return a vector having N components.

    -   t_v specifies a list of the time values at which to integrate the system.  The first value corresponds
        to the initial condition, so the length of t_v must be at least 1.  The timesteps are computed as the
        difference between successive elements.  The timesteps can be negative; see
        https://en.wikipedia.org/wiki/Symplectic_integrator#A_second-order_example

    -   c_v and d_v should be arrays of numbers, having the same length, each summing to 1.
    
        The length of c_v (which should be the same as the length of d_v) is the order
        of the integrator.  The arrays c_v and d_v define the specific integrator (e.g.
        symplectic Euler, Verlet, Ruth 3rd order, Ruth 4th order, etc).

    -   If return_intermediates is False, then the intermediate steps for this particular integrator will be
        included in the integrated_coordinates return value.

    Return values:

    -   integrated_coordinates is a numpy.ndarray having shape (len(t_v),I,2,N), containing the coordinates of
        each integrator step starting with initial_coordinates.  If return_intermediates is False, then I is 1.  If
        return_intermediates is True, then I is 2*order, and the intermediate coordinates (corresponding
        to each c_v[i] and d_v[i] update) are also stored.
    """

    assert np.shape(initial_coordinates)[0] == 2
    assert len(t_v) >= 1
    assert len(c_v) == len(d_v), ''

    N = np.shape(initial_coordinates)[1]
    T = len(t_v)
    order = len(c_v)
    if return_intermediates:
        I = 2*order
    else:
        I = 1

    # Create the return value
    integrated_coordinates = np.ndarray((T,I,2,N), dtype=initial_coordinates.dtype)
    # Create a buffer for intermediate coordinates
    current_coordinates = np.copy(initial_coordinates)

    # Create slices to address the q and p components of current_coordinates.
    q = current_coordinates[0,:]
    p = current_coordinates[1,:]

    for step_index,timestep in enumerate(np.diff(t_v)):
        integrated_coordinates[step_index,0,:,:] = current_coordinates
        # Iterate over (c,d) pairs and perform the update steps.
        for internal_step_index,(c,d) in enumerate(zip(c_v,d_v)):
            q += timestep*c*dK_dp(p)
            if return_intermediates:
                integrated_coordinates[step_index,2*internal_step_index+1,:,:] = current_coordinates
            p -= timestep*d*dV_dq(q)
            if return_intermediates and 2*internal_step_index+2 < I:
                integrated_coordinates[step_index,2*internal_step_index+2,:,:] = current_coordinates

    if not return_intermediates:
        integrated_coordinates[T-1,0,:,:] = current_coordinates
    else:
        for j in range(I):
            integrated_coordinates[T-1,j,:,:] = current_coordinates
        # # This may or may not work -- want to assign same current_coordinates value to all elements of [step_index,:] slice
        # integrated_coordinates[step_index,:,:,:] = current_coordinates

    return integrated_coordinates

def symplectic_euler_method_split_hamiltonian (*, initial_coordinates, dK_dp, dV_dq, t_v, return_intermediates=False):
    return symplectic_integrator_split_hamiltonian(
        initial_coordinates=initial_coordinates,
        dK_dp=dK_dp,
        dV_dq=dV_dq,
        t_v=t_v,
        c_v=[1.0],
        d_v=[1.0],
        return_intermediates=return_intermediates
    )

def symplectic_verlet_method_split_hamiltonian (*, initial_coordinates, dK_dp, dV_dq, t_v, return_intermediates=False):
    return symplectic_integrator_split_hamiltonian(
        initial_coordinates=initial_coordinates,
        dK_dp=dK_dp,
        dV_dq=dV_dq,
        t_v=t_v,
        c_v=[0.0, 1.0],
        d_v=[0.5, 0.5],
        return_intermediates=return_intermediates
    )

def symplectic_ruth3_method_split_hamiltonian (*, initial_coordinates, dK_dp, dV_dq, t_v, return_intermediates=False):
    return symplectic_integrator_split_hamiltonian(
        initial_coordinates=initial_coordinates,
        dK_dp=dK_dp,
        dV_dq=dV_dq,
        t_v=t_v,
        c_v=[1.0, -2.0/3.0, 2.0/3.0],
        d_v=[-1.0/24.0, 0.75, 7.0/24.0],
        return_intermediates=return_intermediates
    )

def symplectic_ruth4_method_split_hamiltonian (*, initial_coordinates, dK_dp, dV_dq, t_v, return_intermediates=False):
    cbrt_2 = 2.0**(1.0/3.0)
    b = 2.0 - cbrt_2
    c_0 = c_3 = 0.5/b
    c_1 = c_2 = 0.5*(1.0 - cbrt_2)/b
    d_0 = d_2 = 1.0/b
    d_1 = -cbrt_2/b
    d_3 = 0.0
    return symplectic_integrator_split_hamiltonian(
        initial_coordinates=initial_coordinates,
        dK_dp=dK_dp,
        dV_dq=dV_dq,
        t_v=t_v,
        c_v=[c_0, c_1, c_2, c_3],
        d_v=[d_0, d_1, d_2, d_3],
        return_intermediates=return_intermediates
    )

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
            I = qp_v.shape[1]

            qp_v_reshaped = qp_v.reshape(T*I, 2)
            H_v_reshaped = H_v.reshape(T*I)

            axis = axis_v[0]
            axis.set_title('{0} : phase space'.format(result_name))
            axis.set_aspect('equal')
            axis.plot(qp_v_reshaped[:,0], qp_v_reshaped[:,1])

            axis = axis_v[1]
            axis.set_title('{0} : Hamiltonian'.format(result_name))
            # axis.plot(t_v, H_v_reshaped)
            axis.plot(H_v_reshaped)

            # sqd_v = np.sum(np.square(qp_v_reshaped - qp_v_reshaped[0]), axis=-1)

            # axis = axis_v[2]
            # axis.set_title('{0} : sq dist from initial'.format(result_name))
            # axis.semilogy(t_v, sqd_v)

        # t_v = np.linspace(0.0, 10.0, 5000)
        t_v = np.linspace(0.0, 30.0, 1500)
        qp_0 = np.array([[np.pi/2.0], [0.0]])
        return_intermediates = False

        add_result('standard odeint', t_v, scipy.integrate.odeint(Pendulum.X_H, qp_0.reshape((2,)), t_v).reshape(len(t_v),1,2,Pendulum.N))
        add_result('symplectic Euler', t_v, symplectic_euler_method_split_hamiltonian(initial_coordinates=qp_0, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, t_v=t_v, return_intermediates=return_intermediates))
        add_result('symplectic Verlet', t_v, symplectic_verlet_method_split_hamiltonian(initial_coordinates=qp_0, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, t_v=t_v, return_intermediates=return_intermediates))
        add_result('symplectic Ruth3', t_v, symplectic_ruth3_method_split_hamiltonian(initial_coordinates=qp_0, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, t_v=t_v, return_intermediates=return_intermediates))
        add_result('symplectic Ruth4', t_v, symplectic_ruth4_method_split_hamiltonian(initial_coordinates=qp_0, dK_dp=Pendulum.dK_dp, dV_dq=Pendulum.dV_dq, t_v=t_v, return_intermediates=return_intermediates))

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
