""" Python Standard Library """
import copy
from operator import itemgetter
""" Third Party Imports """
""" Local Imports """


class ODEFlowTerm:
    def __init__(self, from_cmpt_idx, to_cmpt_idx):
        self.from_cmpt_idx = from_cmpt_idx
        self.to_cmpt_idx = to_cmpt_idx

    def flow_val(self, t_int, y):
        pass

    def add_to_linear_matrix(self, matrix, t_int):
        pass

    def add_to_constant_vector(self, vector, t_int):
        pass

    def add_to_nonlinear_matrices(self, matrices, t_int):
        pass

    def deepcopy(self):
        return self.build(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    @classmethod
    def build(cls, from_cmpt_idx, to_cmpt_idx, coef_by_t=None, scale_by_cmpts_idxs=None,
              scale_by_cmpts_coef_by_t=None, constant_by_t=None):
        if constant_by_t is not None:
            return ConstantODEFlowTerm(from_cmpt_idx, to_cmpt_idx, constant_by_t=constant_by_t)
        elif scale_by_cmpts_coef_by_t is not None:
            return WeightedScaledODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t,
                                             scale_by_cmpts_idxs=scale_by_cmpts_idxs,
                                             scale_by_cmpts_coef_by_t=scale_by_cmpts_coef_by_t)
        elif scale_by_cmpts_idxs is not None:
            return ScaledODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t,
                                     scale_by_cmpts_idxs=scale_by_cmpts_idxs)
        else:
            return LinearODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t)


class ConstantODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, constant_by_t):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.constant_by_t = constant_by_t

    def flow_val(self, t, y):
        return self.constant_by_t[t]

    def add_to_constant_vector(self, vector, t_int):
        vector[self.from_cmpt_idx] -= self.constant_by_t[t_int]
        vector[self.to_cmpt_idx] += self.constant_by_t[t_int]


class LinearODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.coef_by_t = coef_by_t

    def flow_val(self, t_int, y):
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int]

    def add_to_linear_matrix(self, matrix, t_int):
        if self.coef_by_t[t_int] != 0:
            matrix[self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int]
            matrix[self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int]


class ScaledODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t, scale_by_cmpts_idxs):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.coef_by_t = coef_by_t
        self.scale_by_cmpts_idxs = sorted(scale_by_cmpts_idxs)

    def flow_val(self, t_int, y):
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(itemgetter(*self.scale_by_cmpts_idxs)(y))

    def add_to_nonlinear_matrices(self, matrices, t_int):
        if self.coef_by_t[t_int] != 0:
            matrices[tuple(self.scale_by_cmpts_idxs)][self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int]
            matrices[tuple(self.scale_by_cmpts_idxs)][self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int]


class WeightedScaledODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t, scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t=None):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.coef_by_t = coef_by_t
        self.scale_by_cmpts_idxs = scale_by_cmpts_idxs
        self.scale_by_cmpts_coef_by_t = scale_by_cmpts_coef_by_t

    def flow_val(self, t_int, y):
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(
            a * b for a, b in zip(itemgetter(*self.scale_by_cmpts_idxs)(y), self.scale_by_cmpts_coef_by_t[t_int]))

    def add_to_nonlinear_matrices(self, matrices, t_int):
        scale_by_cmpts_coef_by_t = self.scale_by_cmpts_coef_by_t[t_int] if self.scale_by_cmpts_coef_by_t else [1] * len(self.scale_by_cmpts_idxs)
        for scale_by_cmpt_idx, scale_by_cmpt_coef in zip(self.scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t):
            matrices[scale_by_cmpt_idx][self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int] * scale_by_cmpt_coef
            matrices[scale_by_cmpt_idx][self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int] * scale_by_cmpt_coef
