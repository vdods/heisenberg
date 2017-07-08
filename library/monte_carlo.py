import matplotlib.pyplot as plt
import numpy as np
from . import random_radial

class MonteCarlo:
    def __init__ (self, *, obj, initial_parameters, inner_radius, outer_radius, rng_seed, embedding=None):
        self.shape                              = initial_parameters.shape
        self.rng                                = np.random.RandomState(rng_seed)
        self.obj                                = obj
        self.inner_radius                       = inner_radius
        self.outer_radius                       = outer_radius
        self.parameter_history_v                = [initial_parameters]
        self.embedding                          = embedding
        self.obj_history_v                      = [obj(self.__embedded(initial_parameters))]
        self.obj_min                            = self.obj_history_v[-1]
        if self.embedding is None:
            self.embedded_parameter_history_v   = self.parameter_history_v
        else:
            self.embedded_parameter_history_v   = [self.__embedded(initial_parameters)]

    def __embedded (self, parameters):
        if self.embedding is None:
            return parameters
        else:
            return self.embedding(parameters)

    def random_perturbation (self):
        # return self.stddev*self.rng.randn(*self.shape)
        return random_radial.random_point_with_uniform_radial_exponent_distribution(shape=self.shape, inner_radius=self.inner_radius, outer_radius=self.outer_radius, rng=self.rng)

    def compute_next_step (self):
        previous_parameter = self.parameter_history_v[-1]
        previous_parameter_embedding = None if self.embedding is None else self.embedded_parameter_history_v[-1]
        random_perturbation = self.random_perturbation()
        current_parameter = previous_parameter + random_perturbation
        current_parameter_embedding = self.__embedded(current_parameter)

        obj_candidate = self.obj(current_parameter_embedding)
        # Return current_parameter if the obj was lower.
        if obj_candidate < self.obj_min:
            print('-------------------------------- distance of update: {0:e}'.format(np.linalg.norm(random_perturbation)))
            self.obj_min = obj_candidate
            self.parameter_history_v.append(current_parameter)
            if self.embedding is not None:
                self.embedded_parameter_history_v.append(current_parameter_embedding)
            self.obj_history_v.append(obj_candidate)
        # Otherwise return the previous parameter
        else:
            self.parameter_history_v.append(previous_parameter)
            if self.embedding is not None:
                self.embedded_parameter_history_v.append(previous_parameter_embedding)
            self.obj_history_v.append(self.obj_history_v[-1])

        return self.parameter_history_v[-1]

if __name__ == '__main__':
    def test_quadratic ():
        def obj (X):
            return 0.5*np.sum(np.square(X))

        X = np.array([0.5, 0.9])
        opt = MonteCarlo(obj=obj, initial_parameters=X, inner_radius=1.0e-10, outer_radius=1.0e-1, rng_seed=12345)
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

        filename = 'monte_carlo.png'
        plt.savefig(filename)
        print('wrote "{0}"'.format(filename))

    def test_quadratic_with_constraint ():
        def obj (X):
            return 0.5*np.sum(np.square(X))

        def embedding (theta):
            return np.array([np.cos(theta)+0.2, np.sin(theta)+0.7])

        X = np.array([0.5])

        opt = MonteCarlo(obj=obj, initial_parameters=X, inner_radius=1.0e-10, outer_radius=1.0e-1, rng_seed=12345, embedding=embedding)
        for _ in range(10000):
            opt.compute_next_step()

        # For convenience
        X_v = np.copy(opt.parameter_history_v)
        emb_X_v = np.copy(opt.embedded_parameter_history_v)

        fig,axis_vv = plt.subplots(1, 3, squeeze=False, figsize=(30,20))

        axis = axis_vv[0][0]
        axis.set_title('optimization curve (embedding)')
        axis.set_aspect('equal')
        axis.axhline(0, color='black')
        axis.axvline(0, color='black')
        axis.plot(emb_X_v[:,0], emb_X_v[:,1])

        axis = axis_vv[0][1]
        axis.set_title('optimization curve (parameters)')
        axis.plot(X_v)

        axis = axis_vv[0][2]
        axis.set_title('objective history')
        axis.semilogy(opt.obj_history_v)

        filename = 'monte_carlo.embedded.png'
        plt.savefig(filename)
        print('wrote "{0}"'.format(filename))

    test_quadratic()
    test_quadratic_with_constraint()
