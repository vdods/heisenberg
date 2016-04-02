import numpy as np

class Controller:
    """
    Design Notes
    ------------
    The learning rate for gradient descent significantly impacts its quality and performance.
    Given a function and an initial point in its domain, and probably some regularity conditions
    on that function, there exists some learning rate alpha such that discrete steps by
    alpha times the gradient at each step point.

    Computing that theoretical learning rate is nontrivial.  The goal here is to develop an
    adaptive scheme for empirically determining the learning rate.

    The idea is to examine the value of the objective function with respect to the sequence of
    gradient descent steps.  If the learning rate is too low, the objective function will
    descrease slowly, and convergence will take too long (non-asymptotic decrease in the
    short-term, i.e. in as long as a human is likely to want to wait).  If the learning rate
    is too high, then the objective function may occasionally increase.  Furthermore, it is
    desired to have a learning rate that is as high as possible, so that convergence is as
    fast as possible.

    One idea is to track the history of the objective function.  If there are increases in
    its value in recent history, then the learning rate must be decreased.  On the other hand,
    if the objective function is strictly decreasing, then its convergence rate can be analyzed;
    the ratio of the most recent delta to the overall delta of the objective function can be used
    to determine if/when to increase the learning rate.

    Perhaps when the learning rate is changed, the gradient descent should start over at the
    initial point.  This would be done to attempt to avoid ever escaping the basin of the
    current local min.  Otherwise if a reset is not desired, then only the history should be
    cleared, so that it can be used to determine if/when to change the learning rate.

    Implementation Notes
    --------------------
    This class is a bit hacky and hasn't worked as well as hoped -- probably because the
    rules for changing the learning rate are ad-hoc and not particularly well-designed.
    """
    def __init__ (self, initial_learning_rate):
        self.learning_rate = initial_learning_rate
        self.objective_function_history_v = []
        self.learning_rate_too_high_v = []
        self.learning_rate_too_low_v = []

    def record_objective_function_value (self, objective_function_value):
        self.objective_function_history_v.append(objective_function_value)
        if len(self.objective_function_history_v) > 30: # TODO: figure out a good bound
            # delta_v = np.diff(self.objective_function_history_v)
            # assert np.all(delta_v[:-1] < 0), 'all deltas up until possibly now should be negative'
            most_recent_delta = np.diff(self.objective_function_history_v[-2:])
            # If we've seen an increase, then decrease the learning rate.
            if most_recent_delta >= 0:
                self.learning_rate_too_high_v.append(self.learning_rate)
                self.learning_rate *= 0.9 # Arbitrary for now
                self.objective_function_history_v = []
                return True # The history was reset.
            # Otherwise check the convergence rate
            else:
                overall_delta = self.objective_function_history_v[-1] - self.objective_function_history_v[0]
                delta_ratio = most_recent_delta / overall_delta
                # If the convergence is too slow, increase the learning rate
                if delta_ratio > 1.0/60.0: # Arbitrary for now
                    self.learning_rate_too_low_v.append(self.learning_rate)
                    # if len(self.learning_rate_too_high_v) > 0:
                    #     self.learning_rate = np.sqrt(self.learning_rate*np.min(self.learning_rate_too_high_v))
                    # else:
                    #     self.learning_rate *= 1.1 # Should be less than 1.0 / 0.9
                    self.learning_rate *= 1.1 # Should be less than 1.0 / 0.9
                    self.objective_function_history_v = []
                    return True # The history was reset.
        return False # The history was not reset.

