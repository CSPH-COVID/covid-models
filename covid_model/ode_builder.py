import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
from time import perf_counter
import scipy.integrate as spi
import scipy.sparse as spsp
# from blist import blist
from operator import itemgetter
from itertools import count
import math
import numbers


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

    @classmethod
    def build(cls, from_cmpt_idx, to_cmpt_idx, coef_by_t=None, scale_by_cmpts_idxs=None, scale_by_cmpts_coef_by_t=None, constant_by_t=None, pool_cmpt_idxs=None):
        if pool_cmpt_idxs is not None:
            return ConstantFromPoolODEFlowTerm(from_cmpt_idx, to_cmpt_idx, constant_by_t=constant_by_t, pool_cmpt_idxs=pool_cmpt_idxs)
        elif constant_by_t is not None:
            return ConstantODEFlowTerm(from_cmpt_idx, to_cmpt_idx, constant_by_t=constant_by_t)
        elif scale_by_cmpts_coef_by_t is not None:
            return WeightedScaledODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t, scale_by_cmpts_idxs=scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t=scale_by_cmpts_coef_by_t)
        elif scale_by_cmpts_idxs is not None:
            return ScaledODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t, scale_by_cmpts_idxs=scale_by_cmpts_idxs)
        else:
            return LinearODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t)


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
        self.scale_by_cmpt_idxs = sorted(scale_by_cmpts_idxs)

    def flow_val(self, t_int, y):
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(itemgetter(*self.scale_by_cmpt_idxs)(y))

    def add_to_nonlinear_matrices(self, matrices, t_int):
        if self.coef_by_t[t_int] != 0:
            matrices[tuple(self.scale_by_cmpt_idxs)][self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int]
            matrices[tuple(self.scale_by_cmpt_idxs)][self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int]


class WeightedScaledODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t, scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t=None):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.coef_by_t = coef_by_t
        self.scale_by_cmpt_idxs = scale_by_cmpts_idxs
        self.scale_by_cmpts_coef_by_t = scale_by_cmpts_coef_by_t

    def flow_val(self, t_int, y):
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(
            a * b for a, b in zip(itemgetter(*self.scale_by_cmpt_idxs)(y), self.scale_by_cmpts_coef_by_t[t_int]))

    def add_to_nonlinear_matrices(self, matrices, t_int):
        scale_by_cmpts_coef_by_t = self.scale_by_cmpts_coef_by_t[t_int] if self.scale_by_cmpts_coef_by_t else [1] * len(self.scale_by_cmpt_idxs)
        for scale_by_cmpt_idx, scale_by_cmpt_coef in zip(self.scale_by_cmpt_idxs, scale_by_cmpts_coef_by_t):
            matrices[scale_by_cmpt_idx][self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int] * scale_by_cmpt_coef
            matrices[scale_by_cmpt_idx][self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int] * scale_by_cmpt_coef


class ConstantODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, constant_by_t):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.constant_by_t = constant_by_t

    def flow_val(self, t, y):
        return self.constant_by_t[t]

    def add_to_constant_vector(self, vector, t_int):
        vector[self.from_cmpt_idx] -= self.constant_by_t[t_int]
        vector[self.to_cmpt_idx] += self.constant_by_t[t_int]


# class ConstantFromPoolODEFlowTerm(ConstantODEFlowTerm):
#     def __init__(self, from_cmpt_idx, to_cmpt_idx, constant_by_t, pool_cmpt_idxs=None):
#         ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx, constant_by_t=constant_by_t)
#         self.from_cmpt_pool_idxs = pool_cmpt_idxs if pool_cmpt_idxs is not None else [from_cmpt_idx]
#
#     def flow_val(self, t, y):
#         if y[self.from_cmpt_idx] == 0:
#             return 0
#         else:
#             return self.constant_by_t[t] * y[self.from_cmpt_idx] / sum(itemgetter(*self.from_cmpt_pool_idxs)(y))


class ODEBuilder:
    """
    Parameters
    ----------
    trange : array-like
        Indicate the t-values at which the ODE should be evaluated.
        Note that for an ODE of the form dy/dt = f(t, y), this class will
        only compute f(t, y) at values in trange; i.e. it does not support
        f(t, y) being continuous over t.

    attributes : OrderedDict
        Dictionary of attributes that will be used to define compartments.
        One compartment will be created for every unique combination of
        attributes. For example, for attributes...
            {"seir": ["S", "I", "R"], "age": ["under-65", "over-65"]}
        ... there would be 6 compartments:
            [("S", "under-65"), ("I", "under-65"), ("R", "under-65"),
            ("S", "over-65"), ("S", "over-65"), ("S", "over-65")]
        Note that you should provide an OrderedDict here, since the order
        of the compartment tuples is dependent on the order of the keys.

    params : dict
        Parameter lookup dictionary of the form...
            {
                t0: {
                    cmpt0: {param0: val000, param1: val001, ...},
                    cmpt1: {param0: val010, param1: val011, ...},
                    ...
                },
                t1: {
                    cmpt0: {param0: val100, param1: val101, ...},
                    cmpt1: {param0: val110, param1: val111, ...},
                    ...
                },
                ...
            }

    param_attr_names : array-like
        The attribute levels by which paramaters will be allowed to vary.
        The compartment definitions in params should match the attribute
        levels in param_attr_levels.

    """
    def __init__(self, trange, attributes: OrderedDict, params=None, param_attr_names=None):
        self.trange = trange

        self.attributes = attributes
        self.attr_names = list(self.attributes.keys())
        self.compartments_as_index = pd.MultiIndex.from_product(attributes.values(), names=attributes.keys())
        self.compartments = list(self.compartments_as_index)
        self.cmpt_idx_lookup = pd.Series(index=self.compartments_as_index, data=range(len(self.compartments_as_index))).to_dict()
        self.length = len(self.cmpt_idx_lookup)

        self.param_attr_names = list(param_attr_names if param_attr_names is not None else self.attr_names)
        self.param_compartments = list(set(tuple(attr_val for attr_val, attr_name in zip(cmpt, self.attr_names) if attr_name in self.param_attr_names) for cmpt in self.compartments))

        self.params = {t: {pcmpt: {} for pcmpt in self.param_compartments} for t in self.trange}
        self.terms = []

    @property
    def params_as_df(self):
        return pd.concat({t: pd.DataFrame.from_dict(p, orient='index') for t, p in self.params.items()}).rename_axis(index=['t'] + list(self.param_attr_names))

    def attr_level(self, attr_name):
        return self.attr_names.index(attr_name)

    def attr_param_level(self, attr_name):
        return self.param_attr_names.index(attr_name)

    def does_cmpt_have_attrs(self, cmpt, attrs, is_param_cmpts=False):
        return all(cmpt[self.attr_param_level(attr_name) if is_param_cmpts else self.attr_level(attr_name)] == attr_val for attr_name, attr_val in attrs.items())

    def filter_cmpts_by_attrs(self, attrs, is_param_cmpts=False):
        return [cmpt for cmpt in (self.param_compartments if is_param_cmpts else self.compartments) if self.does_cmpt_have_attrs(cmpt, attrs, is_param_cmpts)]

    def set_param(self, param, val=None, attrs=None, trange=None, mult=None, pow=None, except_attrs=None):
        if val is not None:
            def apply(t, cmpt, param):
                self.params[t][cmpt][param] = val
        elif mult is not None:
            def apply(t, cmpt, param):
                self.params[t][cmpt][param] *= mult
        elif pow is not None:
            def apply(t, cmpt, param):
                self.params[t][cmpt][param] = self.params[t][cmpt][param] ** pow
        else:
            raise ValueError('Must provide val, mult, or pow.')
        if type(val if val is not None else mult if mult is not None else val) not in (int, float, np.float64):
            raise TypeError(f'Parameter value (or multiplier) must be numeric; {val if val is not None else mult} is {type(val if val is not None else mult)}')
        if trange is None:
            actual_trange = self.trange
        else:
            actual_trange = set(self.trange).intersection(trange)
        cmpts = self.param_compartments
        if attrs:
            cmpts = self.filter_cmpts_by_attrs(attrs, is_param_cmpts=True)
        if except_attrs:
            cmpts = [cmpt for cmpt in cmpts if cmpt not in self.filter_cmpts_by_attrs(except_attrs, is_param_cmpts=True)]
        for cmpt in cmpts:
            for t in actual_trange:
                apply(t, cmpt, param)

    def calc_coef_by_t(self, coef, cmpt, other_cmpts=None):

        if len(cmpt) > len(self.param_attr_names):
            param_cmpt = tuple(attr for attr, level in zip(cmpt, self.attr_names) if level in self.param_attr_names)
        else:
            param_cmpt = cmpt

        if isinstance(coef, dict):
            return {t: coef[t] if t in coef.keys() else 0 for t in self.trange}
        elif callable(coef):
            return {t: coef(t) for t in self.trange}
        elif isinstance(coef, str):
            if coef == '1':
                coef_by_t = {t: 1 for t in self.trange}
            else:
                coef_by_t = {}
                expr = parse_expr(coef)
                relevant_params = [str(s) for s in expr.free_symbols]
                param_cmpts_by_param = {**{param: param_cmpt for param in relevant_params}, **(other_cmpts if other_cmpts else {})}
                if len(relevant_params) == 1 and coef == relevant_params[0]:
                    coef_by_t = {t: self.params[t][param_cmpt][coef] for t in self.trange}
                else:
                    func = sym.lambdify(relevant_params, expr)
                    for t in self.trange:
                        coef_by_t[t] = func(**{param: self.params[t][param_cmpts_by_param[param]][param] for param in relevant_params})
            return coef_by_t
        else:
            return {t: coef for t in self.trange}

    def reset_ode(self):
        self.terms = []

    def get_terms_by_cmpt(self, from_cmpt, to_cmpt):
        return [term for term in self.terms if term.from_cmpt_idx == self.cmpt_idx_lookup[from_cmpt] and term.to_cmpt_idx == self.cmpt_idx_lookup[to_cmpt]]

    def get_term_indices_by_attr(self, from_attrs, to_attrs):
        return [i for i, term in enumerate(self.terms) if self.does_cmpt_have_attrs(self.compartments[term.from_cmpt_idx], from_attrs) and self.does_cmpt_have_attrs(self.compartments[term.to_cmpt_idx], to_attrs)]

    def get_terms_by_attr(self, from_attrs, to_attrs):
        return [self.terms[i] for i in self.get_term_indices_by_attr(from_attrs, to_attrs)]

    def reset_terms(self, from_attrs, to_attrs):
        for i in sorted(self.get_term_indices_by_attr(from_attrs, to_attrs), reverse=True):
            del self.terms[i]

    def add_flow(self, from_cmpt, to_cmpt, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None, pool_cmpts=None):
        if len(from_cmpt) < len(self.attributes.keys()):
            raise ValueError(f'Origin compartment `{from_cmpt}` does not have the right number of attributes.')
        if len(to_cmpt) < len(self.attributes.keys()):
            raise ValueError(f'Destination compartment `{to_cmpt}` does not have the right number of attributes.')
        if scale_by_cmpts is not None:
            for cmpt in scale_by_cmpts:
                if len(cmpt) < len(self.attributes.keys()):
                    raise ValueError(f'Scaling compartment `{cmpt}` does not have the right number of attributes.')

        if coef is not None:
            if scale_by_cmpts_coef:
                coef_by_t_lookup = {c: self.calc_coef_by_t(c, to_cmpt) for c in set(scale_by_cmpts_coef)}
                coef_by_t_ld = [coef_by_t_lookup[c] for c in scale_by_cmpts_coef]
                coef_by_t_dl = {t: [dic[t] for dic in coef_by_t_ld] for t in self.trange}
            else:
                coef_by_t_dl = None

        self.terms.append(ODEFlowTerm.build(
            from_cmpt_idx=self.cmpt_idx_lookup[from_cmpt],
            to_cmpt_idx=self.cmpt_idx_lookup[to_cmpt],
            coef_by_t=self.calc_coef_by_t(coef, from_cmpt),  # switched BACK to setting parameters use the TO cmpt
            scale_by_cmpts_idxs=[self.cmpt_idx_lookup[cmpt] for cmpt in scale_by_cmpts] if scale_by_cmpts is not None else None,
            scale_by_cmpts_coef_by_t=coef_by_t_dl if scale_by_cmpts is not None else None,
            constant_by_t=self.calc_coef_by_t(constant, to_cmpt) if constant is not None else None,
            pool_cmpt_idxs=[self.cmpt_idx_lookup[pool_cmpt] for pool_cmpt in pool_cmpts] if pool_cmpts is not None else None))

    def add_flows_by_attr(self, from_attrs, to_attrs, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None, from_pool=False):
        from_cmpts = self.filter_cmpts_by_attrs(from_attrs)
        for from_cmpt in from_cmpts:
            to_cmpt_list = list(from_cmpt)
            for attr_name, new_attr_val in to_attrs.items():
                to_cmpt_list[self.attr_level(attr_name)] = new_attr_val
            to_cmpt = tuple(to_cmpt_list)
            self.add_flow(from_cmpt, to_cmpt, coef=coef, scale_by_cmpts=scale_by_cmpts,
                          scale_by_cmpts_coef=scale_by_cmpts_coef, constant=constant, pool_cmpts=from_cmpts if from_pool else None)

    def compile(self):
        self.linear_matrix = {t: spsp.lil_matrix((self.length, self.length)) for t in self.trange}
        self.nonlinear_matrices = {t: defaultdict(lambda: spsp.lil_matrix((self.length, self.length))) for t in self.trange}
        self.constant_vector = {t: np.zeros(self.length) for t in self.trange}

        for term in self.terms:
            for t in self.trange:
                term.add_to_linear_matrix(self.linear_matrix[t], t)
                term.add_to_nonlinear_matrices(self.nonlinear_matrices[t], t)
                term.add_to_constant_vector(self.constant_vector[t], t)

        # convert to CSR for better performance
        for t in self.trange:
            self.linear_matrix[t] = self.linear_matrix[t].tocsr()
            for k, v in self.nonlinear_matrices[t].items():
                self.nonlinear_matrices[t][k] = v.tocsr()

    def y0_from_dict(self, y0_dict):
        y0 = [0]*self.length
        for cmpt, n in y0_dict.items():
            y0[self.cmpt_idx_lookup[cmpt]] = n
        return y0

    def ode(self, t, y):
        dy = [0] * self.length
        t_int = min(math.floor(t), len(self.trange) - 1)

        dy += self.linear_matrix[t_int].dot(y)
        for scale_by_cmpt_idxs, matrix in self.nonlinear_matrices[t_int].items():
            dy += sum(itemgetter(*scale_by_cmpt_idxs)(y)) * matrix.dot(y)

        dy += self.constant_vector[t_int]

        return dy

    def solve_ode(self, y0_dict, method='RK45'):
        self.solution = spi.solve_ivp(
            fun=self.ode,
            t_span=[min(self.trange), max(self.trange)],
            y0=self.y0_from_dict(y0_dict),
            t_eval=self.trange,
            # jac_sparsity=self.jac_sparsity,
            # jac=self.jacobian,
            method=method)
        if not self.solution.success:
            raise RuntimeError(f'ODE solver failed with message: {self.solution.message}')
        self.solution_y = np.transpose(self.solution.y)
        self.solution_ydf = pd.concat([self.y_to_series(self.solution_y[t]) for t in self.trange], axis=1, keys=self.trange, names=['t']).transpose()

    def y_to_series(self, y):
        return pd.Series(index=self.compartments_as_index, data=y)

    def solution_sum(self, group_by_attr_levels=None):
        if group_by_attr_levels:
            return self.solution_ydf.groupby(group_by_attr_levels, axis=1).sum()

    def mean_params_as_df(self, group_by_attr_levels=None):
        params = self.params_as_df.rename_axis('param', axis=1)

        if group_by_attr_levels is None:
            stacked_solution = self.solution_ydf.stack(level=list(self.attr_names))
            expanded_params = params
            expanded_solution = pd.concat({col: stacked_solution for col in params.columns}, axis=1, names=['param'])
        else:
            non_group_by_attr_levels = [attr_name for attr_name in self.attr_names if attr_name not in group_by_attr_levels]
            stacked_solution = self.solution_ydf.stack(level=list(non_group_by_attr_levels)).rename_axis(group_by_attr_levels, axis=1)
            expanded_params = pd.concat({col: params for col in stacked_solution.columns}, axis=1, names=[*group_by_attr_levels])
            expanded_solution = pd.concat({col: stacked_solution for col in params.columns}, axis=1, names=['param']).reorder_levels([*group_by_attr_levels, 'param'], axis=1)

        return (expanded_params * expanded_solution).groupby('t').sum() / stacked_solution.groupby('t').sum()