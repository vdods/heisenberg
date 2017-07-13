import numpy as np
from . import orbit_plot
import os
import sys
import time
from . import util
import vorpy
import vorpy.pickle
import vorpy.symplectic_integration

class ShootingMethodObjective:
    def __init__ (self, *, dynamics_context, qp_0, t_max, t_delta, disable_salvage=False):
        self.__dynamics_context         = dynamics_context
        self.alpha                      = dynamics_context.alpha()
        self.beta                       = dynamics_context.beta()
        self.qp_0                       = qp_0
        self.__t_v                      = None
        self.__qp_v                     = None
        self.t_max                      = t_max
        self.t_delta                    = t_delta
        self.__Q_v                      = None
        self.__Q_global_min_index       = None
        self.__t_min                    = None
        self.__objective                = None
        self.__disable_salvage          = disable_salvage
        self.flow_curve_was_salvaged    = False

    def configuration_space_dimension (self):
        return self.__dynamics_context.configuration_space_dimension()

    def flow_curve (self):
        if self.__qp_v is None:
            start_time = time.time() # TODO: Replace with Ticker usage

            t_v = np.arange(0.0, self.t_max, self.t_delta)
            order = 2
            #omega = vorpy.symplectic_integration.nonseparable_hamiltonian.heuristic_estimate_for_omega(delta=self.t_delta, order=order, c=10.0)
            # Want 2*omega*t_delta = pi/2, meaning that omega = pi/(4*t_delta)
            omega = np.pi/(4*self.t_delta)
            assert np.allclose(2*omega*self.t_delta, np.pi/2)
            try:
                qp_v = vorpy.symplectic_integration.nonseparable_hamiltonian.integrate(
                    initial_coordinates=self.qp_0,
                    t_v=t_v,
                    dH_dq=self.__dynamics_context.dH_dq,
                    dH_dp=self.__dynamics_context.dH_dp,
                    order=order,
                    omega=omega
                )
                self.flow_curve_was_salvaged        = False
            except vorpy.symplectic_integration.exceptions.SalvagedResultException as e:
                print('salvaged results from exception encountered in nonseparable_hamiltonian.integrate: {0}'.format(e))
                original_step_count             = len(t_v)
                qp_v                            = e.salvaged_qp_v
                t_v                             = e.salvaged_t_v
                self.flow_curve_was_salvaged    = True

                # Set these here so that this ShootingMethodObjective is fully defined for use in OrbitPlot.plot_curve.
                self.__t_v  = t_v
                self.__qp_v = qp_v

                if not self.__disable_salvage:
                    # TEMP: Plot this salvaged curve in order to diagnose what went wrong
                    curve_description = 'salvaged curve - {0} steps out of {1}'.format(e.salvaged_qp_v.shape[0], original_step_count)
                    op = orbit_plot.OrbitPlot(curve_description_v=[curve_description], quantity_to_plot_v=orbit_plot.default_quantity_to_plot_v)
                    op.plot_curve(curve_description=curve_description, smo=self)
                    op.savefig_and_clear(
                        filename=os.path.join(
                            'heisenberg.custom_plot', # TODO: Specify salvaged result directory
                            'salvaged.obj:{0:.4e}.t_delta:{1:.3e}.t_max:{2:.3e}.ic:{3}.png'.format(
                                self.objective(),
                                self.t_delta,
                                self.t_max,
                                util.ndarray_as_single_line_string(self.qp_0)
                            )
                        )
                    )

            print('integration took {0} seconds'.format(time.time() - start_time))

            self.__t_v  = t_v
            self.__qp_v = qp_v
            assert self.__qp_v is not None
        return self.__qp_v

    def t_v (self):
        if self.__t_v is None:
            self.flow_curve()
            assert self.__t_v is not None
        return self.__t_v

    def Q_v (self):
        if self.__Q_v is None:
            # Let s denote squared distance function s(t) := 1/2 |qp_0 - flow_of_qp_0(t))|^2
            #self.__Q_v = 0.5 * np.sum(np.square(self.flow_curve() - self.qp_0), axis=-1)
            self.__Q_v = vorpy.apply_along_axes(lambda x:0.5*np.sum(np.square(x)), (-2,-1), (self.flow_curve() - self.qp_0,), output_axis_v=(), func_output_shape=())
            assert self.__Q_v is not None
        return self.__Q_v

    def t_min (self):
        if self.__t_min is None:
            self.compute_t_min_and_objective()
            assert self.__t_min is not None
        return self.__t_min

    def objective (self):
        if self.__objective is None:
            self.compute_t_min_and_objective()
            assert self.__objective is not None
        return self.__objective

    def Q_global_min_index (self):
        if self.__Q_global_min_index is None:
            self.compute_t_min_and_objective()
        return self.__Q_global_min_index

    def __call__ (self):
        return self.objective()

    def compute_t_min_and_objective (self):
        t_v                             = self.t_v()
        Q_v                             = self.Q_v()

        local_min_index_v               = [i for i in range(1,len(Q_v)-1) if Q_v[i-1] > Q_v[i] and Q_v[i] < Q_v[i+1]]
        Q_local_min_v                   = [Q_v[i] for i in local_min_index_v]
        try:
            Q_local_min_min_index       = np.argmin(Q_local_min_v)
            self.__Q_global_min_index   = _Q_global_min_index = local_min_index_v[Q_local_min_min_index]
            if True:
                # Fit a quadratic function to the 3 points centered on the argmin in order to have
                # sub-sample accuracy when calculating the objective function value.
                assert 1 <= _Q_global_min_index < len(Q_v)-1
                s                       = slice(_Q_global_min_index-1, _Q_global_min_index+2)
                self.__t_min,self.__objective = util.quadratic_min_time_parameterized(t_v[s], Q_v[s])
                # Some tests show this discrepancy to be on the order of 1.0e-9
                #print('self.__objective - Q_v[_Q_global_min_index] = {0}'.format(self.__objective - Q_v[_Q_global_min_index]))
            else:
                self.__t_min            = t_v[_Q_global_min_index]
                self.__objective        = Q_v[_Q_global_min_index]
        except ValueError:
            # If there was no local min, then declare the objective function value to be NaN
            self.__Q_global_min_index   = None
            self.__t_min                = np.nan
            self.__objective            = np.nan

    def pickle (self, filename):
        # First, ensure everything is computed.
        pickle_data = {
            'alpha':self.alpha,
            'beta':self.beta,
            't_v':self.t_v(),
            'qp_v':self.flow_curve(),
            'Q_v':self.Q_v(),
            'Q_global_min_index':self.Q_global_min_index(),
            't_min':self.t_min(),
            'obj':self.objective(),
        }
        # Not sure if there's a guarantee as to the order the above dict elements are computed, so
        # ensure that self.flow_curve() has been called before assigning the flow_curve_was_salvaged
        # attribute in the dict.
        pickle_data['flow_curve_was_salvaged'] = self.flow_curve_was_salvaged,

        vorpy.pickle.try_to_pickle(data=pickle_data, pickle_filename=filename, log_out=sys.stdout)
        print('wrote to "{0}"'.format(filename))

def evaluate_shooting_method_objective (dynamics_context, qp_0, t_max, t_delta, disable_salvage=False):
    """A utility function for constructing a ShootingMethodObjective instance and evaluating it."""
    print('evaluate_shooting_method_objective; trying qp_0 = {0}'.format(qp_0))
    smo = ShootingMethodObjective(dynamics_context=dynamics_context, qp_0=qp_0, t_max=t_max, t_delta=t_delta, disable_salvage=disable_salvage)
    objective = smo.objective()
    return objective

