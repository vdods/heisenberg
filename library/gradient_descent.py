import numpy as np
import random

class BlindGradientDescent:
    def __init__ (self, obj, Dobj, initial_parameters, learning_rate):
        self.obj = obj
        self.Dobj = Dobj
        self.learning_rate = learning_rate
        self.obj_history_v = [obj(initial_parameters)]
        self.__next_step = np.ndarray((len(initial_parameters),), dtype=float)

    def compute_next_step (self, current_parameters, compute_obj=False):
        step_direction = self.Dobj(current_parameters)
        self.__next_step[:] = current_parameters - self.learning_rate*step_direction
        if compute_obj:
            obj_next_step = self.obj(self.__next_step)
            self.obj_history_v.append(obj_next_step)
        return self.__next_step

class GradientDescent:
    def __init__ (self, obj, Dobj, initial_parameters, learning_rate_v):
        self.obj = obj
        self.Dobj = Dobj
        self.learning_rate_v = learning_rate_v
        if type(self.learning_rate_v) != np.ndarray:
            self.learning_rate_v = np.array(self.learning_rate_v)
        self.obj_history_v = [obj(initial_parameters)]
        self.best_learning_rate_history_v = []

        self.__next_step_v = np.ndarray((len(self.learning_rate_v),len(initial_parameters)), dtype=float)
        self.__obj_next_step_v = np.ndarray((len(self.learning_rate_v),), dtype=float)

    def compute_next_step (self, current_parameters):
        step_direction = self.Dobj(current_parameters)

        # minimizing_obj_next_step = self.obj_history_v[-1]
        # learning_rate_v = self.learning_rate_v
        # while True:
        #     self.__next_step_v[:] = current_parameters - np.einsum('i,j->ij', learning_rate_v, step_direction)
        #     self.__obj_next_step_v[:] = np.apply_along_axis(self.obj, 1, self.__next_step_v)

        #     minimizing_next_step_index = np.argmin(self.__obj_next_step_v)
        #     minimizing_obj_next_step = self.__obj_next_step_v[minimizing_next_step_index]

        #     if minimizing_obj_next_step < self.obj_history_v[-1]:
        #         break

        #     learning_rate_v = 0.5*learning_rate_v

        self.__next_step_v[:] = current_parameters - np.einsum('i,j->ij', self.learning_rate_v, step_direction)
        self.__obj_next_step_v[:] = np.apply_along_axis(self.obj, 1, self.__next_step_v)

        minimizing_next_step_index = np.argmin(self.__obj_next_step_v)
        minimizing_obj_next_step = self.__obj_next_step_v[minimizing_next_step_index]

        if minimizing_obj_next_step >= self.obj_history_v[-1]:
            # Only step on decrease; let the stochastic part try again later.
            self.best_learning_rate_history_v.append(0.0)
            self.obj_history_v.append(self.obj_history_v[-1])
            return current_parameters
        else:
            self.best_learning_rate_history_v.append(self.learning_rate_v[minimizing_next_step_index])
            self.obj_history_v.append(minimizing_obj_next_step)
            return self.__next_step_v[minimizing_next_step_index]

class BatchedStochasticGradientComputer:
    def __init__ (self, Dobj, sample_count, batch_size=None):
        self.Dobj = Dobj
        self.sample_count = sample_count
        if batch_size == None:
            batch_size = sample_count
        self.batch_size = batch_size
        self.sample_index_v = np.arange(sample_count)
        batch_index_slice_v = [
            slice(batch_index*batch_size,(batch_index+1)*batch_size)
            for batch_index in xrange((sample_count+batch_size-1)//batch_size)
        ]
        assert batch_index_slice_v[-1].stop >= sample_count
        self.batch_vv = [self.sample_index_v[batch_index_slice] for batch_index_slice in batch_index_slice_v]
        self.generate_new_batches()

    def generate_new_batches (self):
        if self.batch_size < self.sample_count:
            random.shuffle(self.sample_index_v) # This should affect the contents of self.batch_vv, because they're just views.
        self.current_batch_index = 0

    def eval_gradient_on_current_batch (self, parameters):
        return self.Dobj(parameters, self.batch_vv[self.current_batch_index])

    def go_to_next_batch (self):
        self.current_batch_index += 1
        if self.current_batch_index == len(self.batch_vv):
            self.generate_new_batches()

