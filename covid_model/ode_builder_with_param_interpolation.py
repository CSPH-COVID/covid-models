import math
from operator import itemgetter

from covid_model.ode_builder import ODEBuilder


class ODEBuilderWithParamInterpolation(ODEBuilder):

    def ode(self, t, y):
        dy = [0] * self.length
        t_int = int(t)
        t_prev = self.t_prev_lookup[t_int]
        t_next = self.t_next_lookup[t_int]
        td = (t - t_prev) / (t_next - t_prev)

        # apply linear terms
        linear_matrix = (1 - td) * self.linear_matrix[t_prev] + td * self.linear_matrix[t_next]
        dy += linear_matrix.dot(y)

        # apply non-linear terms
        for (scale_by_cmpt_idxs_prev, matrix_prev), (scale_by_cmpt_idxs_next, matrix_next) in zip(self.nonlinear_matrices[t_prev].items(), self.nonlinear_matrices[t_next].items()):
            nonlinear_multiplier = (1 - td) * self.nonlinear_multiplier[t_prev] + td * self.nonlinear_multiplier[t_next]
            matrix = (1 - td) * matrix_prev + td * matrix_next
            dy += nonlinear_multiplier * sum(itemgetter(*scale_by_cmpt_idxs_prev)(y)) * matrix.dot(y)

        # apply constant terms
        constant_vector = (1 - td) * self.constant_vector[t_prev] + td * self.constant_vector[t_next]
        dy += constant_vector

        return dy
