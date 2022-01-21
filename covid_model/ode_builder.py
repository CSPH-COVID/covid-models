import numpy as np
import pandas as pd
from collections import OrderedDict
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
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t, scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t):
        self.from_cmpt_idx = from_cmpt_idx
        self.to_cmpt_idx = to_cmpt_idx
        self.coef_by_t = coef_by_t
        self.is_scaled_by_cmpts = scale_by_cmpts_idxs is not None and scale_by_cmpts_idxs != []
        self.scale_by_cmpt_idxs = scale_by_cmpts_idxs
        # self.scale_by_cmpts_coef_by_t = scale_by_cmpts_coef_by_t if scale_by_cmpts_coef_by_t is not None else {t: np.array([1]*len(self.scale_by_cmpt_idxs)) for t in range(len(self.coef_by_t.keys()))}
        self.scale_by_cmpts_coef_by_t = scale_by_cmpts_coef_by_t

        self.solution = None
        self.solution_y = None
        self.solution_ydf = None

    # def flow_rate(self, t, y):
    #     if self.scale_by_cmpt_idxs:
    #         if self.scale_by_cmpts_coef_by_t:
    #             return self.coef_by_t[t] * sum(a * b for a, b in zip(itemgetter(*self.scale_by_cmpt_idxs)(y), self.scale_by_cmpts_coef_by_t[t]))
    #         else:
    #             return self.coef_by_t[t] * sum(itemgetter(*self.scale_by_cmpt_idxs)(y))
    #     else:
    #         return self.coef_by_t[t]

    def flow_val(self, t_int, y):
        if self.is_scaled_by_cmpts:
            if self.scale_by_cmpts_coef_by_t:
                return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(a * b for a, b in zip(itemgetter(*self.scale_by_cmpt_idxs)(y), self.scale_by_cmpts_coef_by_t[t_int]))
            else:
                return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(itemgetter(*self.scale_by_cmpt_idxs)(y))
        else:
            # print('MOO')
            # print(y)
            # print(self.from_cmpt_idx)
            # print(y[self.from_cmpt_idx])
            # print(self.coef_by_t)
            # print(t_int)
            # print(self.coef_by_t[t_int])
            # print(y[self.from_cmpt_idx] * self.coef_by_t[t_int])
            return y[self.from_cmpt_idx] * self.coef_by_t[t_int]

    def jacobian(self, t, y):
        jac = spsp.lil_matrix((len(y), len(y)))
        if self.scale_by_cmpt_idxs is None or len(self.scale_by_cmpt_idxs) > 0:
            jac[self.from_cmpt_idx, self.from_cmpt_idx] = -self.coef_by_t[t]
            jac[self.to_cmpt_idx, self.from_cmpt_idx] = self.coef_by_t[t]
        else:
            if self.scale_by_cmpts_coef_by_t is not None:
                for sbc_idx, sbc_coef in zip(self.scale_by_cmpt_idxs, self.scale_by_cmpts_coef_by_t[t]):
                    jac[self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t] * sbc_coef * y[sbc_idx]
                    jac[self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t] * sbc_coef * y[sbc_idx]
                    jac[self.from_cmpt_idx, sbc_idx] = -self.coef_by_t[t] * sbc_coef * y[self.from_cmpt_idx]
                    jac[self.to_cmpt_idx, sbc_idx] = self.coef_by_t[t] * sbc_coef * y[self.from_cmpt_idx]
            else:
                for sbc_idx in self.scale_by_cmpt_idxs:
                    jac[self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t] * y[sbc_idx]
                    jac[self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t] * y[sbc_idx]
                    jac[self.from_cmpt_idx, sbc_idx] = -self.coef_by_t[t] * y[self.from_cmpt_idx]
                    jac[self.to_cmpt_idx, sbc_idx] = self.coef_by_t[t] * y[self.from_cmpt_idx]

        return jac

    def non_linear_jacobian(self, t, y):
        jac = spsp.lil_matrix((len(y), len(y)))

        if self.scale_by_cmpts_coef_by_t is not None:
            for sbc_idx, sbc_coef in zip(self.scale_by_cmpt_idxs, self.scale_by_cmpts_coef_by_t[t]):
                jac[self.from_cmpt_idx, sbc_idx] = self.coef_by_t[t] * sbc_coef * y[self.from_cmpt_idx]
                jac[self.to_cmpt_idx, sbc_idx] = self.coef_by_t[t] * sbc_coef * y[self.from_cmpt_idx]
        else:
            for sbc_idx in self.scale_by_cmpt_idxs:
                jac[self.from_cmpt_idx, sbc_idx] = self.coef_by_t[t] * y[self.from_cmpt_idx]
                jac[self.to_cmpt_idx, sbc_idx] = self.coef_by_t[t] * y[self.from_cmpt_idx]

        return jac


class ConstantODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, constant_by_t):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx, coef_by_t=None,
                             scale_by_cmpts_idxs=None, scale_by_cmpts_coef_by_t=None)

        self.constant_by_t = constant_by_t

    def flow_val(self, t, y):
        return self.constant_by_t[t]


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
        self.jac_sparsity = spsp.lil_matrix((self.length, self.length), dtype=int)
        self.jac_fixed_by_t = {}

        # self.jacobians = {t: np.zeros((self.length, self.length)) for t in self.trange}
        # self.linear_jacobians_by_t = {}
        self.terms = []

    # @property
    # def length(self):
    #     return len(self.cmpt_idx_lookup)

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

    def set_param(self, name, val=None, attrs=None, trange=None, mult=None):
        if type(val if val is not None else mult) not in (int, float, np.float64):
            raise TypeError(f'Parameter value (or multiplier) must be numeric; {val if val is not None else mult} is {type(val if val is not None else mult)}')
        if trange is None:
            actual_trange = self.trange
        else:
            actual_trange = set(self.trange).intersection(trange)
        for cmpt in self.filter_cmpts_by_attrs(attrs, is_param_cmpts=True) if attrs else self.param_compartments:
            for t in actual_trange:
                if val is not None:
                    self.params[t][cmpt][name] = val
                else:
                    self.params[t][cmpt][name] *= mult

    def calc_coef_by_t(self, coef, cmpt):

        if len(cmpt) > len(self.param_attr_names):
            param_cmpt = tuple(attr for attr, level in zip(cmpt, self.attr_names) if level in self.param_attr_names)
        else:
            param_cmpt = cmpt

        if isinstance(coef, dict):
            return {t: coef[t] if t in coef.keys() else 0 for t in self.trange}
        elif callable(coef):
            return {t: coef(t) for t in self.trange}
        elif isinstance(coef, str):
            coef_by_t = {}
            expr = parse_expr(coef)
            relevant_params = [str(s) for s in expr.free_symbols]
            func = sym.lambdify(relevant_params, expr)
            for t in self.trange:
                coef_by_t[t] = func(**{k: v for k, v in self.params[t][param_cmpt].items() if k in relevant_params})
            return coef_by_t
        else:
            return {t: coef for t in self.trange}

    def reset_ode(self):
        self.terms = []

    def reset_terms(self, from_attrs, to_attrs):
        to_delete = []
        for i, term in enumerate(self.terms):
            if self.does_cmpt_have_attrs(self.compartments[term.from_cmpt_idx], from_attrs) and self.does_cmpt_have_attrs(self.compartments[term.to_cmpt_idx], to_attrs):
                to_delete.append(i)

        for i in sorted(to_delete, reverse=True):
            del self.terms[i]

    def add_flow(self, from_cmpt, to_cmpt, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None):
        self.jac_sparsity[self.cmpt_idx_lookup[from_cmpt], self.cmpt_idx_lookup[from_cmpt]] = 1
        self.jac_sparsity[self.cmpt_idx_lookup[to_cmpt], self.cmpt_idx_lookup[from_cmpt]] = 1
        if scale_by_cmpts:
            for scale_by_cmpt in scale_by_cmpts:
                self.jac_sparsity[self.cmpt_idx_lookup[from_cmpt], self.cmpt_idx_lookup[scale_by_cmpt]] = 1
                self.jac_sparsity[self.cmpt_idx_lookup[to_cmpt], self.cmpt_idx_lookup[scale_by_cmpt]] = 1
        if len(from_cmpt) < len(self.attributes.keys()):
            raise ValueError(f'Origin compartment `{from_cmpt}` does not have the right number of attributes.')
        if len(to_cmpt) < len(self.attributes.keys()):
            raise ValueError(f'Destination compartment `{to_cmpt}` does not have the right number of attributes.')
        if scale_by_cmpts is not None:
            for cmpt in scale_by_cmpts:
                if len(cmpt) < len(self.attributes.keys()):
                    raise ValueError(f'Scaling compartment `{cmpt}` does not have the right number of attributes.')

        if coef is not None:
            self.terms.append(ODEFlowTerm(
                from_cmpt_idx=self.cmpt_idx_lookup[from_cmpt],
                to_cmpt_idx=self.cmpt_idx_lookup[to_cmpt],
                coef_by_t=self.calc_coef_by_t(coef, from_cmpt),
                scale_by_cmpts_idxs=[self.cmpt_idx_lookup[cmpt] for cmpt in scale_by_cmpts] if scale_by_cmpts is not None else [],
                scale_by_cmpts_coef_by_t=pd.DataFrame([self.calc_coef_by_t(c, from_cmpt) for c in scale_by_cmpts_coef]).to_dict(orient='list') if scale_by_cmpts_coef is not None else None))

        if constant is not None:
            self.terms.append(ConstantODEFlowTerm(
                from_cmpt_idx=self.cmpt_idx_lookup[from_cmpt],
                to_cmpt_idx=self.cmpt_idx_lookup[to_cmpt],
                constant_by_t=self.calc_coef_by_t(constant, from_cmpt)
            ))

    def precalc_jacobians(self):
        for t in self.trange:
            self.jac_fixed_by_t[t] = spsp.lil_matrix((self.length, self.length))
            for term in self.terms:
                if term.scale_by_cmpt_idxs is None or len(term.scale_by_cmpt_idxs) == 0:
                    self.jac_fixed_by_t[t][term.from_cmpt_idx, term.from_cmpt_idx] -= term.coef_by_t[t]
                    self.jac_fixed_by_t[t][term.to_cmpt_idx, term.from_cmpt_idx] += term.coef_by_t[t]

    def jacobian(self, t, y):
        t_int = min(np.floor(t), len(self.trange) - 1)
        jac = spsp.lil_matrix((self.length, self.length))
        for term in self.terms:
            if term.scale_by_cmpt_idxs is None or len(term.scale_by_cmpt_idxs) == 0:
                jac[term.from_cmpt_idx, term.from_cmpt_idx] -= term.coef_by_t[t_int]
                jac[term.to_cmpt_idx, term.from_cmpt_idx] += term.coef_by_t[t_int]
            else:
                for sbc_idx, sbc_coef in zip(term.scale_by_cmpt_idxs, term.scale_by_cmpts_coef_by_t[t_int] if term.scale_by_cmpts_coef_by_t is not None else [1] * len(term.scale_by_cmpt_idxs)):
                    jac[term.from_cmpt_idx, term.from_cmpt_idx] -= term.coef_by_t[t_int] * sbc_coef * y[sbc_idx]
                    jac[term.to_cmpt_idx, term.from_cmpt_idx] += term.coef_by_t[t_int] * sbc_coef * y[sbc_idx]
                    jac[term.from_cmpt_idx, sbc_idx] -= -term.coef_by_t[t_int] * sbc_coef * y[term.from_cmpt_idx]
                    jac[term.to_cmpt_idx, sbc_idx] += term.coef_by_t[t_int] * sbc_coef * y[term.from_cmpt_idx]

        return jac

    def y0_from_dict(self, y0_dict):
        y0 = [0]*self.length
        for cmpt, n in y0_dict.items():
            y0[self.cmpt_idx_lookup[cmpt]] = n
        return y0

    def ode(self, t, y):
        dy = [0] * self.length
        t_int = min(math.floor(t), len(self.trange) - 1)

        for term in self.terms:
            val = term.flow_val(t_int, y)
            # if term.is_scaled_by_cmpts:
            #     if term.scale_by_cmpts_coef_by_t:
            #         val = y[term.from_cmpt_idx] * term.coef_by_t[t_int] * sum(a * b for a, b in
            #                                                                zip(itemgetter(*term.scale_by_cmpt_idxs)(y),
            #                                                                    term.scale_by_cmpts_coef_by_t[t_int]))
            #     else:
            #         val = y[term.from_cmpt_idx] * term.coef_by_t[t_int] * sum(itemgetter(*term.scale_by_cmpt_idxs)(y))
            # else:
            #     val = y[term.from_cmpt_idx] * term.coef_by_t[t_int]

            dy[term.from_cmpt_idx] -= val
            dy[term.to_cmpt_idx] += val

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
