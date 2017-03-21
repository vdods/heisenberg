import matplotlib.pyplot as plt
import numpy as np

class MonteCarlo:
    def __init__ (self, obj, initial_parameters, stddev, rng_seed): # TODO: Better radial distribution with accumulation point at origin
        self.shape = initial_parameters.shape
        self.rng = np.random.RandomState(rng_seed)
        self.obj = obj
        self.stddev = stddev
        self.parameter_history_v = [initial_parameters]
        self.obj_history_v = [obj(initial_parameters)]
        self.obj_min = self.obj_history_v[-1]

    def random_perturbation (self):
        return self.stddev*self.rng.randn(*self.shape)

    def compute_next_step (self):
        previous_parameter = self.parameter_history_v[-1]
        current_parameter = previous_parameter + self.random_perturbation()

        obj_candidate = self.obj(current_parameter)
        # Return current_parameter if the obj was lower.
        if obj_candidate < self.obj_min:
            self.obj_min = obj_candidate
            self.parameter_history_v.append(current_parameter)
            self.obj_history_v.append(obj_candidate)
        # Otherwise return the previous parameter
        else:
            self.parameter_history_v.append(previous_parameter)
            self.obj_history_v.append(self.obj_history_v[-1])

        return self.parameter_history_v[-1]

if __name__ == '__main__':
    # test

    def Q (X):
        return 0.5*np.sum(np.square(X))

    X = np.array([0.5, 0.9])
    opt = MonteCarlo(Q, X, 0.001, 12345)
    for _ in range(10000):
        opt.compute_next_step()

    # For convenience
    X_v = np.copy(opt.parameter_history_v)

    fig,axis_vv = plt.subplots(1, 2, squeeze=False, figsize=(20,20))

    axis = axis_vv[0][0]
    axis.set_title('optimization curve')
    axis.set_aspect('equal')
    axis.axhline(0, color='black')
    axis.axvline(0, color='black')
    axis.plot(X_v[:,0], X_v[:,1])

    axis = axis_vv[0][1]
    axis.set_title('objective history')
    axis.semilogy(opt.obj_history_v)

    plt.savefig('monte_carlo.png')
