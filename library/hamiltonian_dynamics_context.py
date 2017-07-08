import abc
import numpy as np

class HamiltonianDynamicsContext(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def configuration_space_dimension (cls):
        pass

    @classmethod
    @abc.abstractmethod
    def H (cls, qp):
        """Evaluates the Hamiltonian on the (2,N)-shaped (q,p) coordinate."""
        pass

    @classmethod
    @abc.abstractmethod
    def dH_dq (cls, q, p):
        """Evaluates the partial of H with respect to q on the (2,N)-shaped (q,p) coordinate.  Returns a (N,)-vector."""
        pass

    @classmethod
    @abc.abstractmethod
    def dH_dp (cls, q, p):
        """Evaluates the partial of H with respect to q on the (2,N)-shaped (q,p) coordinate.  Returns a (N,)-vector."""
        pass

    @classmethod
    def X_H (cls, qp):
        """
        Computes the Hamiltonian vector field on coordinates qp (with shape (2,N)), returning the same shape.

        \omega^-1 * dH (i.e. the symplectic gradient of H) is the hamiltonian vector field for this system.

        If the tautological one-form on the cotangent bundle is

            tau := p dq

        then the symplectic form is

            omega := -dtau = -dq wedge dp

        which, e.g. in the coordinates (q_0, q_1, p_0, p_1), has the matrix

            [  0  0 -1  0 ]
            [  0  0  0 -1 ]
            [  1  0  0  0 ]
            [  0  1  0  0 ],

        or in matrix notation, with I denoting the 2x2 identity matrix,

            [  0 -I ]
            [  I  0 ],

        having inverse

            [  0  I ]
            [ -I  0 ].

        With dH:

            dH = dH/dq * dq + dH/dp * dp,    (here, dH/dq denotes the partial of H w.r.t. q)

        or expressed in coordinates as

            [ dH/dq ]
            [ dH/dp ]

        it follows that the sympletic gradient of H is

            dH/dp * dq - dH/dq * dp

        or expressed in coordinates as

            [  dH/dp ]
            [ -dH/dq ].

        The equation defining the flow for this vector field is

            dq/dt =  dH/dp
            dp/dt = -dH/dq,

        which is Hamilton's equations.
        """
        q = coordinates[0,:]
        p = coordinates[1,:]
        # This is the symplectic gradient of H.
        retval = np.vstack((cls.dH_dp(q,p), -cls.dH_dq(q,p)))
        assert retval.shape[0] == 2
        return retval

    @classmethod
    def phase_space_dimension (cls):
        return 2*cls.configuration_space_dimension()

