import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import scipy.integrate as spi
import scipy.sparse as spsp
from operator import itemgetter
import math
import copy


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
    def build(cls, from_cmpt_idx, to_cmpt_idx, coef_by_t=None, scale_by_cmpts_idxs=None, scale_by_cmpts_coef_by_t=None, constant_by_t=None):
        if constant_by_t is not None:
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


class ConstantODEFlowTerm(ODEFlowTerm):
    def __init__(self, from_cmpt_idx, to_cmpt_idx, constant_by_t):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.constant_by_t = constant_by_t

    def flow_val(self, t, y):
        return self.constant_by_t[t]

    def add_to_constant_vector(self, vector, t_int):
        vector[self.from_cmpt_idx] -= self.constant_by_t[t_int]
        vector[self.to_cmpt_idx] += self.constant_by_t[t_int]


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

    param_attr_names : array-like
        The attribute levels by which paramaters will be allowed to vary.
        The compartment definitions in params should match the attribute
        levels in param_attr_levels.

    """
    def __init__(self, base_ode_builder=None, trange=None, attributes: OrderedDict = None, param_attr_names=None, deepcopy_params=True):
        # TODO: self.terms is actually not used anymore, since we have the matrices; should it be removed or adjusted?

        self.trange = trange if trange is not None else base_ode_builder.trange
        self.t_prev_lookup = None
        self.t_next_lookup = None
        self.build_t_lookups()
        self.attributes = attributes if attributes is not None else base_ode_builder.attributes
        self.param_attr_names = param_attr_names if param_attr_names is not None else attributes.keys() if attributes is not None else base_ode_builder.param_attr_names

        self.attr_names = list(self.attributes.keys())
        self.compartments_as_index = pd.MultiIndex.from_product(attributes.values(), names=attributes.keys())
        self.compartments = list(self.compartments_as_index)
        self.cmpt_idx_lookup = pd.Series(index=self.compartments_as_index, data=range(len(self.compartments_as_index))).to_dict()
        self.length = len(self.cmpt_idx_lookup)
        self.param_compartments = list(set(tuple(attr_val for attr_val, attr_name in zip(cmpt, self.attr_names) if attr_name in self.param_attr_names) for cmpt in self.compartments))

        self.params = None
        self.terms = None
        self.linear_matrix = None
        self.nonlinear_matrices = None
        self.constant_vector = None
        self.nonlinear_multiplier = None

        self.solution = None
        self.solution_y = None
        self.solution_ydf = None

        if base_ode_builder is not None:
            if deepcopy_params:
                self.params = {t: {cmpt: params.copy() for cmpt, params in params_by_cmpt.items()} for t, params_by_cmpt in base_ode_builder.params.items() for t in self.trange}
            else:
                self.params = {t: params_by_cmpt for t, params_by_cmpt in base_ode_builder.params.items() for t in self.trange}
            self.terms = {term.deepcopy() for term in base_ode_builder.terms}
            self.linear_matrix = {t: m.copy() for t, m in base_ode_builder.linear_matrix.items() if t in self.trange}
            self.nonlinear_matrices = {t: m.copy() for t, m in base_ode_builder.nonlinear_matrices.items() if t in self.trange}
            self.constant_vector = {t: m.copy() for t, m in base_ode_builder.constant_vector.items() if t in self.trange}
            self.nonlinear_multiplier = base_ode_builder.nonlinear_multiplier.copy()
        else:
            self.params = {t: {pcmpt: {} for pcmpt in self.param_compartments} for t in self.trange}
            self.reset_ode()

    # the t-values at which the model results will be evaluated in outputs (and for fitting)
    @property
    def t_eval(self):
        return range(min(self.trange), max(self.trange))

    # return the parameters as a dataframe with t and compartments as index and parameters as columns
    @property
    def params_as_df(self):
        return pd.concat({t: pd.DataFrame.from_dict(self.params[self.t_prev_lookup[t]], orient='index') for t in self.t_eval}).rename_axis(index=['t'] + list(self.param_attr_names))

    def build_t_lookups(self):
        self.t_prev_lookup = {t_int: max(t for t in self.trange if t <= t_int) for t_int in range(min(self.trange), max(self.trange))}
        self.t_prev_lookup[max(self.trange)] = self.t_prev_lookup[max(self.trange) - 1]
        self.t_next_lookup = {t_int: min(t for t in self.trange if t > t_int) for t_int in range(min(self.trange), max(self.trange))}
        self.t_next_lookup[max(self.trange)] = self.t_next_lookup[max(self.trange) - 1]

    # get the level associated with a given attribute name
    # e.g. if attributes are ['seir', 'age', 'variant'], the level of 'age' is 1 and the level of 'variant' is 2
    def attr_level(self, attr_name):
        return self.attr_names.index(attr_name)

    # get the level associated with a param attribute
    def param_attr_level(self, attr_name):
        return self.param_attr_names.index(attr_name)

    # check if a cmpt matches a dictionary of attributes
    def does_cmpt_have_attrs(self, cmpt, attrs, is_param_cmpts=False):
        return all(
            cmpt[self.param_attr_level(attr_name) if is_param_cmpts else self.attr_level(attr_name)]
            in ([attr_val] if isinstance(attr_val, str) else attr_val)
            for attr_name, attr_val in attrs.items())

    # return compartments that match a dictionary of attributes
    def filter_cmpts_by_attrs(self, attrs, is_param_cmpts=False):
        return [cmpt for cmpt in (self.param_compartments if is_param_cmpts else self.compartments) if self.does_cmpt_have_attrs(cmpt, attrs, is_param_cmpts)]

    # return the "first" compartment that matches a dictionary of attributes, with "first" determined by attribute order
    def get_default_cmpt_by_attrs(self, attrs):
        return tuple(attrs[attr_name] if attr_name in attrs.keys() else attr_list[0] for attr_name, attr_list in self.attributes.items())

    # set a parameter (if val is provided; otherwise apply a multiplier or exponent)
    def set_param(self, param, val=None, attrs=None, trange=None, mult=None, pow=None, except_attrs=None, desc=None):
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
            for attr in [attrs] if isinstance(attrs, dict) else attrs:
                cmpts = self.filter_cmpts_by_attrs(attr, is_param_cmpts=True)
        if except_attrs:
            for except_attr in [except_attrs] if isinstance(except_attrs, dict) else except_attrs:
                cmpts = [cmpt for cmpt in cmpts if cmpt not in self.filter_cmpts_by_attrs(except_attr, is_param_cmpts=True)]
        for cmpt in cmpts:
            for t in actual_trange:
                apply(t, cmpt, param)

    # set the "non-linear multiplier" which is a scalar (for each t value) that will scale all non-linear flows
    # used for changing TC without rebuilding all the matrices
    def set_nonlinear_multiplier(self, mult, trange=None):
        trange = trange if trange is not None else self.trange
        for t in trange:
            self.nonlinear_multiplier[t] = mult

    def get_param(self, param, attrs=None, trange=None):
        if trange is None:
            actual_trange = self.trange
        else:
            actual_trange = set(self.trange).intersection(trange)
        cmpt_list = self.filter_cmpts_by_attrs(attrs, is_param_cmpts=True) if attrs else self.param_compartments
        return [(cmpt, [self.params[t][cmpt][param] for t in actual_trange]) for cmpt in cmpt_list]

    # takes a symbolic equation, and looks up variable names in params to provide a computed output for each t in trange
    def calc_coef_by_t(self, coef, cmpt, other_cmpts=None, trange=None):

        trange = trange if trange is not None else self.trange

        if len(cmpt) > len(self.param_attr_names):
            param_cmpt = tuple(attr for attr, level in zip(cmpt, self.attr_names) if level in self.param_attr_names)
        else:
            param_cmpt = cmpt

        if isinstance(coef, dict):
            return {t: coef[t] if t in coef.keys() else 0 for t in trange}
        elif callable(coef):
            return {t: coef(t) for t in trange}
        elif isinstance(coef, str):
            if coef == '1':
                coef_by_t = {t: 1 for t in trange}
            else:
                coef_by_t = {}
                expr = parse_expr(coef)
                relevant_params = [str(s) for s in expr.free_symbols]
                param_cmpts_by_param = {**{param: param_cmpt for param in relevant_params}, **(other_cmpts if other_cmpts else {})}
                if len(relevant_params) == 1 and coef == relevant_params[0]:
                    coef_by_t = {t: self.params[t][param_cmpt][coef] for t in trange}
                else:
                    func = sym.lambdify(relevant_params, expr)
                    for t in trange:
                        coef_by_t[t] = func(**{param: self.params[t][param_cmpts_by_param[param]][param] for param in relevant_params})
            return coef_by_t
        else:
            return {t: coef for t in trange}

    # assign default values to matrices
    def reset_ode(self):
        self.terms = []
        self.linear_matrix = {t: spsp.lil_matrix((self.length, self.length)) for t in self.trange}
        self.nonlinear_matrices = {t: defaultdict(lambda: spsp.lil_matrix((self.length, self.length))) for t in self.trange}
        self.constant_vector = {t: np.zeros(self.length) for t in self.trange}
        self.nonlinear_multiplier = {}

    # get all terms that refer to flow from one specific compartment to another
    def get_terms_by_cmpt(self, from_cmpt, to_cmpt):
        return [term for term in self.terms if term.from_cmpt_idx == self.cmpt_idx_lookup[from_cmpt] and term.to_cmpt_idx == self.cmpt_idx_lookup[to_cmpt]]

    # get the indices of all terms that refer to flow from one specific compartment to another
    def get_term_indices_by_attr(self, from_attrs, to_attrs):
        return [i for i, term in enumerate(self.terms) if self.does_cmpt_have_attrs(self.compartments[term.from_cmpt_idx], from_attrs) and self.does_cmpt_have_attrs(self.compartments[term.to_cmpt_idx], to_attrs)]

    # get the terms that refer to flow from compartments with a set of attributes to compartments with another set of attributes
    def get_terms_by_attr(self, from_attrs, to_attrs):
        return [self.terms[i] for i in self.get_term_indices_by_attr(from_attrs, to_attrs)]

    # remove terms that refer to flow from compartments with a set of attributes to compartments with another set of attributes
    def reset_terms(self, from_attrs, to_attrs):
        for i in sorted(self.get_term_indices_by_attr(from_attrs, to_attrs), reverse=True):
            del self.terms[i]

    # add a flow term, and add new flow to ODE matrices
    def add_flow(self, from_cmpt, to_cmpt, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None):
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

        term = ODEFlowTerm.build(
            from_cmpt_idx=self.cmpt_idx_lookup[from_cmpt],
            to_cmpt_idx=self.cmpt_idx_lookup[to_cmpt],
            coef_by_t=self.calc_coef_by_t(coef, to_cmpt),  # switched BACK to setting parameters use the TO cmpt
            scale_by_cmpts_idxs=[self.cmpt_idx_lookup[cmpt] for cmpt in
                                 scale_by_cmpts] if scale_by_cmpts is not None else None,
            scale_by_cmpts_coef_by_t=coef_by_t_dl if scale_by_cmpts is not None else None,
            constant_by_t=self.calc_coef_by_t(constant, to_cmpt) if constant is not None else None)

        self.terms.append(term)

        # add term to matrices
        for t in self.trange:
            term.add_to_linear_matrix(self.linear_matrix[t], t)
            term.add_to_nonlinear_matrices(self.nonlinear_matrices[t], t)
            term.add_to_constant_vector(self.constant_vector[t], t)

    # add multipler flows, from all compartments with from_attrs, to compartments that match the from compartments, but replacing attributes as designated in to_attrs
    # e.g. from {'seir': 'S', 'age': '0-19'} to {'seir': 'E'} will be a flow from susceptible 0-19-year-olds to exposed 0-19-year-olds
    def add_flows_by_attr(self, from_attrs, to_attrs, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None):
        from_cmpts = self.filter_cmpts_by_attrs(from_attrs)
        for from_cmpt in from_cmpts:
            to_cmpt_list = list(from_cmpt)
            for attr_name, new_attr_val in to_attrs.items():
                to_cmpt_list[self.attr_level(attr_name)] = new_attr_val
            to_cmpt = tuple(to_cmpt_list)
            self.add_flow(from_cmpt, to_cmpt, coef=coef, scale_by_cmpts=scale_by_cmpts,
                          scale_by_cmpts_coef=scale_by_cmpts_coef, constant=constant)

    # convert ODE matrices to CSR format, to (massively) improve performance
    def compile(self):
        for t in self.trange:
            self.linear_matrix[t] = self.linear_matrix[t].tocsr()
            for k, v in self.nonlinear_matrices[t].items():
                self.nonlinear_matrices[t][k] = v.tocsr()

    # create a y0 vector with all values as 0, except those designated in y0_dict
    def y0_from_dict(self, y0_dict):
        y0 = [0]*self.length
        for cmpt, n in y0_dict.items():
            y0[self.cmpt_idx_lookup[cmpt]] = n
        return y0

    # ODE step forward
    def ode(self, t, y):
        dy = [0] * self.length
        t_int = self.t_prev_lookup[math.floor(t)]

        # apply linear terms
        dy += (self.linear_matrix[t_int]).dot(y)

        # apply non-linear terms
        for scale_by_cmpt_idxs, matrix in self.nonlinear_matrices[t_int].items():
            dy += self.nonlinear_multiplier[t_int] * sum(itemgetter(*scale_by_cmpt_idxs)(y)) * (matrix).dot(y)

        # apply constant terms
        dy += self.constant_vector[t_int]

        return dy

    # solve ODE using scipy.solve_ivp, and put solution in solution_y and solution_ydf
    # TODO: try Julia ODE package, to improve performance
    def solve_ode(self, y0_dict, method='RK45'):
        t_eval = range(min(self.trange), max(self.trange))
        self.solution = spi.solve_ivp(
            fun=self.ode,
            t_span=[min(self.trange), max(self.trange)],
            y0=self.y0_from_dict(y0_dict),
            t_eval=t_eval,
            method=method)
        if not self.solution.success:
            raise RuntimeError(f'ODE solver failed with message: {self.solution.message}')
        self.solution_y = np.transpose(self.solution.y)
        self.solution_ydf = pd.concat([self.y_to_series(self.solution_y[t]) for t in t_eval], axis=1, keys=t_eval, names=['t']).transpose()

    # convert y-array to series with compartment attributes as multiindex
    def y_to_series(self, y):
        return pd.Series(index=self.compartments_as_index, data=y)

    # return solution grouped by group_by_attr_levels
    def solution_sum(self, group_by_attr_levels):
        return self.solution_ydf.groupby(group_by_attr_levels, axis=1).sum()