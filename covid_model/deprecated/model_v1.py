import json
import math
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import pyswarms as ps
from sqlalchemy import MetaData
from datetime import datetime
import itertools
from data_imports import ExternalHosps, ExternalVacc
from utils import *
from collections import OrderedDict


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel:
    # the variables in the differential equation
    vars = ['S', 'E', 'I', 'Ih', 'A', 'R', 'RA', 'V', 'Vxd', 'D']

    transitions = [
        ('s', 'e', 'rel_inf_prob'),
        ('e', 'i', 'pS'),
        ('i', 'h', 'hosp'),
        ('i', 'd', 'dnh'),
        ('h', 'd', 'dh')]

    groups = ['0-19', '20-39', '40-64', '65+']

    # the starting date of the model
    datemin = dt.datetime(2020, 1, 24)

    def __init__(self, tslices, efs=None, fit_id=None, engine=None):
        self.tslices = list(tslices)

        # build global parameters from params file, creating dict lookups for easy access
        self.engine = engine
        self.gparams = None
        self.gparams_lookup = None

        # vaccines
        self.vacc_rate_df = None
        self.vacc_immun_df = None
        self.vacc_trans_mults_df = None
        self.vacc_prevalence_df = None

        # variants
        self.variant_prevalence_df = None

        # transmission control parameters
        self.efs = efs
        self.ef_by_t = None

        # the var values for the solution; these get populated when self.solve_seir is run
        self.solution = None
        self.solution_y = None
        self.solution_ydf_full = None

        # used to connect up with the matching fit in the database
        self.fit_id = fit_id

    def prepped_duplicate(self):
        new_model = CovidModel(self.tslices.copy(), efs=self.efs.copy(), fit_id=self.fit_id, engine=self.engine)
        new_model.gparams = self.gparams.copy()
        new_model.gparams_lookup = self.gparams_lookup.copy()
        new_model.variant_prevalence_df = self.variant_prevalence_df.copy()
        new_model.vacc_rate_df = self.vacc_rate_df.copy()
        new_model.vacc_immun_df = self.vacc_immun_df.copy()
        new_model.ef_by_t = self.ef_by_t.copy()

    # a model must be prepped before it can be run; if any params EXCEPT the efs (i.e. TC) change, it must be re-prepped
    def prep(self, params=None, vacc_proj_params=None, vacc_immun_params='input/vacc_immun_params.json'):
        # prep general parameters (gparams_lookup)
        if self.gparams_lookup is None or params is not None:
            self.set_gparams(params if params is not None else 'input/params.json')
            # prep variants (self.variant_prevalence_df and updates to self.gparams_lookup)
            if 'variants' in self.gparams and self.gparams['variants']:
                self.set_variant_params(self.gparams['variants'])

        # prep vacc rates, including projections (vacc_rate_df)
        if self.vacc_rate_df is None or vacc_proj_params is not None:
            self.set_vacc_rates(vacc_proj_params if vacc_proj_params is not None else json.load(open('input/vacc_proj_params.json'))['current trajectory'])

        # prep vaccine immunity (vacc_immun_df and updates to self.gparams_lookup)
        # vaccine immunity is dependent on gparams and vacc proj, so if anything has been re-prepped, vaccine immunity will need to be re-prepped
        if self.vacc_immun_df is None or vacc_immun_params is not None or vacc_proj_params is not None or params is not None:
            self.set_vacc_immun(vacc_immun_params if vacc_immun_params is not None else 'input/vacc_immun_params.json')
            self.apply_vacc_multipliers(vacc_immun_params if vacc_immun_params is not None else 'input/vacc_immun_params.json')

        # prep efs (ef_by_t)
        if self.efs is not None:
            self.set_ef_by_t(self.efs)

    def set_ef_from_db(self, fit_id, extend=True):
        fit = CovidModelFit.from_db(self.engine, fit_id)
        tslices = fit.tslices
        efs = fit.efs
        if extend and self.tmax > tslices[-1]:
            tslices.append(self.tmax)
            efs.append(efs[-1])

        self.tslices = tslices
        self.efs = efs
        self.set_ef_by_t(self.efs)

    # create using on a fit that was run previously or manually inserted into the database
    @staticmethod
    def from_fit(conn, fit_id):
        fit = CovidModelFit.from_db(conn, fit_id)
        model = CovidModel(tslices=fit.tslices, engine=conn)
        model.set_gparams(fit.model_params)
        model.set_ef_by_t(fit.efs)

        return model

    # coerce a "y", a list of length (num of vars)*(num of groups), into a dataframe with dims (num of groups)x(num of vars)
    @classmethod
    def y_to_df(cls, y):
        return pd.DataFrame(np.array(y).reshape(len(cls.groups), len(cls.vars)), columns=cls.vars, index=cls.groups)

    @property
    def solution_ydf(self):
        df = self.solution_ydf_full.copy()
        # df['V'] = df['V'] + df['Vxd']
        # df = df.drop(columns='Vxd')
        return df

    # handy properties for the beginning t, end t, and the full range of t values
    @property
    def tmin(self): return self.tslices[0]

    @property
    def tmax(self): return self.tslices[-1]

    @property
    def trange(self): return range(self.tmin, self.tmax)

    @property
    def daterange(self): return pd.date_range(self.datemin, periods=len(self.trange))

    # sum the solution vars to create a single df with total values across all groups
    @property
    def solution_ydf_summed(self):
        return self.solution_ydf.groupby('t').sum()

    @property
    def solution_dydf(self):
        gpdf = pd.DataFrame.from_dict(self.gparams_lookup, orient='index')
        print(gpdf)
        exit()
        dydf = pd.DataFrame(index=self.solution_ydf.index)
        dydf_yesterday = self.solution_ydf.groupby('group').shift(1)
        # print(dydf_yesterday['Ih'].reset_index().apply(lambda x: print(x), axis=1)) #self.gparams_lookup[x.reset_index()['t']][x.reset_index()['group']]['lhos'], axis=1))
        dydf['Ih'] = self.solution_ydf['Ih'] - dydf_yesterday['Ih'] + dydf_yesterday['Ih'] / dydf_yesterday['Ih'].apply(lambda x: self.gparams_lookup[x['t']][x['group']], axis=1)
        return dydf

    # new exposures by day by group
    @property
    def new_exposures(self):
        return self.solution_ydf['E'] / self.gparams['alpha']

    # estimated reproduction number (length of infection * new_exposures / current_infections
    @property
    def re_estimates(self):
        infect_duration = 1 / self.gparams['gamma']
        infected = self.solution_ydf['I'].groupby('t').sum().shift(3)
        return infect_duration * self.new_exposures.groupby('t').sum() / infected

    # calc the observed TC by applying the variant multiplier to the base TC
    @property
    def obs_ef_by_t(self):
        return {t: sum(1 - (1 - self.ef_by_t[t]) * self.gparams_lookup[t][group]['rel_inf_prob'] for group in self.groups) / len(self.groups) for t in self.trange}

    @property
    def obs_ef_by_slice(self):
        oef_by_t = self.obs_ef_by_t
        return [np.array([oef_by_t[t] for t in range(self.tslices[i], self.tslices[i+1])]).mean() for i in range(len(self.tslices) - 1)]

    # build dataframe containing vaccine first-dose rates by day by group by vaccine
    def set_vacc_rates(self, proj_params):
        proj_params_dict = proj_params if isinstance(proj_params, dict) else json.load(open(proj_params))

        # self.vacc_rate_df = get_vaccinations(self.engine, proj_params_dict, from_date=self.datemin, proj_to_date=self.daterange.max(), groupN=self.gparams['groupN'])
        self.vacc_rate_df = ExternalVacc(self.engine, t0_date=self.datemin, fill_to_date=max(self.daterange)).fetch('input/past_and_projected_vaccinations.csv', proj_params=proj_params_dict, groupN=self.gparams['groupN'])
        # self.vacc_rate_df.index = self.vacc_rate_df.index.set_levels((self.vacc_rate_df.index.unique(0) - self.datemin).days, level=0).set_names('t', level=0)
        self.vacc_rate_df['cumu'] = self.vacc_rate_df.groupby(['group', 'vacc'])['rate'].cumsum()

    # build dataframe containing the gain and loss of vaccine immunity by day by group
    def set_vacc_immun(self, immun_params):
        immun_params_dict = immun_params if isinstance(immun_params, dict) else json.load(open(immun_params))

        # set immunity gain and loss
        immun_gains = []
        immun_losses = []
        for vacc, vacc_specs in immun_params_dict.items():
            immun_gains += [delay_specs['value'] * self.vacc_rate_df['rate'].xs(vacc, level='vacc', drop_level=False).groupby('group').shift(delay_specs['delay']) for delay_specs in vacc_specs['immun_gain']]
            immun_losses += [delay_specs['value'] * self.vacc_rate_df['rate'].xs(vacc, level='vacc', drop_level=False).groupby('group').shift(delay_specs['delay']) for delay_specs in vacc_specs['immun_loss']]
        self.vacc_immun_df = pd.DataFrame()
        self.vacc_immun_df['immun_gain'] = pd.concat(immun_gains).fillna(0).groupby(['t', 'group']).sum().sort_index()
        self.vacc_immun_df['immun_loss'] = pd.concat(immun_losses).fillna(0).groupby(['t', 'group']).sum().sort_index()

        # add to gparams_lookup
        for t in self.trange:
            for group in self.groups:
                for param in ['immun_gain', 'immun_loss']:
                    self.gparams_lookup[t][group][f'vacc_{param}'] = self.vacc_immun_df.loc[(t, group), param]

    # build dataframe of multipliers and apply them to pS, hosp, dh, and dnh
    def apply_vacc_multipliers(self, immun_params):
        immun_params_dict = immun_params if isinstance(immun_params, dict) else json.load(open(immun_params))

        # build dataframe
        combined_mults = {}
        vacc_prevs = {}
        for t in self.trange:
            for g in self.groups:
                mults, s_prevs = ([], [])
                for vacc, vacc_specs in immun_params_dict.items():
                    for delay_specs in vacc_specs['multipliers']:
                        mult = delay_specs['value'].copy()
                        mult['rel_inf_prob'] += self.gparams['delta_vacc_escape'] * self.variant_prevalence_df.loc[(t, g, 'delta'), 'e_prev']
                        mults.append(mult)
                        s_prevs.append(self.vacc_rate_df.loc[(max(t - delay_specs['delay'], 0), g, vacc), 'cumu'] / self.gparams['groupN'][g])
                combined_mults[(t, g)], vacc_prevs[(t, g)] = calc_multiple_multipliers(self.transitions, mults, s_prevs)

        # create a dataframe of the multipliers for each transition param; drop rel_inf_prob because we're handling that through immun_gain and immun_loss
        self.vacc_trans_mults_df = pd.DataFrame.from_dict(combined_mults, orient='index').drop(columns='rel_inf_prob')
        self.vacc_trans_mults_df.index.rename(['t', 'group'], inplace=True)

        # create a dataframe of the estimated vaccine prevalence for individuals ENTERING each bucket
        # for deaths, this definitely needs to averaged over history, since people stay dead a long time
        self.vacc_prevalence_df = pd.DataFrame.from_dict(vacc_prevs, orient='index')

        # for every transition except s -> e, multiply the transition param by the vacc mult
        params = self.vacc_trans_mults_df.columns
        for t in self.trange:
            for group in self.groups:
                for param in params:
                    self.gparams_lookup[t][group][param] *= self.vacc_trans_mults_df.loc[(t, group), param]

    def write_vacc_to_csv(self, fname):
        df = pd.DataFrame()
        df['first_shot_rate'] = self.vacc_rate_df['rate'].groupby(['t', 'group']).sum().fillna(0)
        df = df.join(self.vacc_immun_df)
        df = df.join(self.vacc_trans_mults_df.rename(columns={col: f'{col}_mult' for col in self.vacc_trans_mults_df.columns}))
        df['first_shot_cumu'] = df['first_shot_rate'].groupby('group').cumsum()
        df['jnj_first_shot_rate'] = self.vacc_rate_df['rate'].xs('jnj', level='vacc')
        df['mrna_first_shot_rate'] = self.vacc_rate_df['rate'].xs('mrna', level='vacc')
        df.to_csv(fname)

    def set_variant_params(self, variant_params):
        dfs = {}
        for variant, specs in variant_params.items():
            var_df = pd.read_csv(specs['theta_file_path'])  # file with at least a col "t" and a col containing variant prevalence
            var_df = var_df.rename(columns={specs['theta_column']: variant})[['t', variant]].set_index('t').rename(columns={variant: 'e_prev'}).astype(float)  # get rid of all columns except t (the index) and the prev value
            if 't_min' in specs.keys():
                var_df['e_prev'].loc[:specs['t_min']] = 0
            mult_df = pd.DataFrame(specs['multipliers'], index=self.groups).rename(columns={col: f'{col}_mult' for col in specs['multipliers'].keys()})
            mult_df.index = mult_df.index.rename('group')
            combined = pd.MultiIndex.from_product([var_df.index, mult_df.index], names=['t', 'group']).to_frame().join(var_df).join(mult_df).drop(columns=['t', 'group'])  # cross join
            dfs[variant] = combined
        df = pd.concat(dfs)
        df.index = df.index.set_names(['variant', 't', 'group']).reorder_levels(['t', 'group', 'variant'])
        df = df.sort_index()

        # fill in future variant prevalence by duplicating the last row
        variant_input_tmax = df.index.get_level_values('t').max()
        if variant_input_tmax < self.tmax:
            # projections = pd.concat({(t, group): df[(variant_input_tmax, group)] for t, group in itertools.product(range(variant_input_tmax + 1), self.groups)})
            projections = pd.concat({t: df.loc[variant_input_tmax] for t in range(variant_input_tmax + 1, self.tmax)})
            df = pd.concat([df, projections]).sort_index()

        # calculate multipliers; run s -> e separately, because we're setting the same variant prevalences in S and E
        df['s_prev'] = df['e_prev']
        df = self.calc_multipliers(df, start_at=0, end_at=0)
        df = self.calc_multipliers(df, start_at=1, add_remaining=False)
        sums = df.groupby(['t', 'group']).sum()
        variant_tmin = sums.index.get_level_values('t').min()
        variant_tmax = sums.index.get_level_values('t').max()
        for t in self.trange:
            if t >= variant_tmin:
                for fr, to, label in self.transitions:
                    for group in self.groups:
                        # if t is greater than variant_tmax, just pull the multiplier at variant_tmax
                        self.gparams_lookup[t][group][label] *= sums.loc[(min(t, variant_tmax), group), f'{label}_flow']
                    if len(set(self.gparams_lookup[t][g][label] for g in self.groups)) == 1:
                        self.gparams_lookup[t][None][label] = self.gparams_lookup[t][self.groups[0]][label]

        self.variant_prevalence_df = df

    # provide a dataframe with [compartment]_prev as the initial prevalence and this function will add the flows necessary to calc the downstream multipliers
    def calc_multipliers(self, df, start_at=0, end_at=10, add_remaining=True):
        if add_remaining:
            remaining = (1.0 - df[[f'{self.transitions[start_at][0]}_prev']].groupby(['t', 'group']).sum())
            remaining['vacc'] = 'none'
            remaining['shot'] = 'none'
            remaining = remaining.set_index(['vacc', 'shot'], append=True)
            df = df.append(remaining)
        df = df.sort_index()
        mult_cols = [col for col in df.columns if col[-5:] == '_mult']
        df[mult_cols] = df[mult_cols].fillna(1.0)
        for fr, to, label in self.transitions[start_at:(end_at+1)]:
            df[f'{label}_flow'] = df[f'{fr}_prev'] * df[f'{label}_mult']
            df[f'{to}_prev'] = df[f'{label}_flow'] / df[f'{label}_flow'].groupby(['t', 'group']).transform(sum)
        return df

    def set_generic_gparams(self, gparams):
        gparams_by_t = {t: get_params(gparams, t) for t in self.trange}
        for t in self.trange:
            for group in self.groups:
                # set "temp" to 1; if there is no "temp_on" parameter or temp_on == False, it will be 1
                self.gparams_lookup[t][group]['temp'] = 1
                self.gparams_lookup[t][group]['rel_inf_prob'] = 1.0
                for k, v in gparams_by_t[t].items():
                    # vaccines and variants are handled separately, so skip
                    if k in ['vaccines', 'variants']:
                        pass
                    # special rules for the "temp" paramater, which is set dynamically based on t
                    elif k == 'temp_on':
                        if v:
                            self.gparams_lookup[t][group]['temp'] = 0.5 * math.cos((t + 45) * 0.017) + 1.5
                    # for all other cases, if it's a dictionary, it should be broken out by group
                    elif type(v) == dict:
                        self.gparams_lookup[t][group][k] = v[group]
                    # if it's not a dict, it should be a single value: just assign that value to all groups
                    else:
                        self.gparams_lookup[t][group][k] = v
            # if all groups have the same value, create a None entry for that param
            self.gparams_lookup[t][None] = dict()
            for k, v in self.gparams_lookup[t][self.groups[0]].items():
                if type(v) != dict and len(set(self.gparams_lookup[t][g][k] for g in self.groups)) == 1:
                    self.gparams_lookup[t][None][k] = v

    # set global parameters and lookup dicts
    def set_gparams(self, params):
        # load gparams
        self.gparams = params if type(params) == dict else json.load(open(params))
        self.gparams_lookup = {t: {g: dict() for g in self.groups} for t in self.trange}
        # build a dictionary of gparams for every (t, group) for convenient access
        self.set_generic_gparams(self.gparams)

    # set ef by slice and lookup dicts
    def set_ef_by_t(self, ef_by_slice):
        self.efs = ef_by_slice
        self.ef_by_t = {t: get_value_from_slices(self.tslices, list(ef_by_slice), t) for t in self.trange}

    # extend the time range with an additional slice; should maybe change how this works to return a new CovidModel instead
    def add_tslice(self, t, ef=None):
        if t <= self.tslices[-2]:
            raise ValueError(f'New tslice (t={t}) must be greater than the second to last tslices (t={self.tslices[-2]}).')
        if t < self.tmax:
            tmax = self.tmax
            self.tslices[-1] = t
            self.tslices.append(tmax)
        else:
            self.tslices.append(t)
        if ef:
            self.efs.append(ef)
            self.set_ef_by_t(self.efs)
        self.fit_id = None

    # this is the rate of flow from S -> E, based on beta, current prevalence, TC, variants, and a bunch of paramaters that should probably be deprecated
    @staticmethod
    def daily_transmission_per_susc(ef, I_total, A_total, rel_inf_prob, N, beta, temp, mask, lamb, siI, ramp, **excess_args):
        return beta * (1 - ef) * rel_inf_prob * (I_total * lamb + A_total) / N

    # the diff eq for a single group; will be called four times in the actual diff eq
    @staticmethod
    def single_group_seir(y, single_group_y, transm_per_susc, vacc_immun_gain, vacc_immun_loss, alpha, gamma, pS, hosp, hlos, dnh, dh, groupN, delta_vacc_escape, delta_share, immune_rate_I, immune_rate_A, dimmuneI=999999, dimmuneA=999999, **excess_args):

        S, E, I, Ih, A, R, RA, V, Vxd, D = single_group_y
        # beta, ef, rel_inf_prob, lamb, N,
        # transm_per_susc = beta * (1 - ef) * rel_inf_prob * (I_total * lamb + A_total) / N

        daily_vacc_per_elig = vacc_immun_gain / (groupN - V - Vxd - Ih - D)

        I2R = I * (gamma * (1 - hosp - dnh)) * immune_rate_I
        I2S = I * (gamma * (1 - hosp - dnh)) * (1 - immune_rate_I)

        Ih2R = (1 - dh) * Ih / hlos * immune_rate_I
        Ih2S = (1 - dh) * Ih / hlos * (1 - immune_rate_I)

        A2RA = A * gamma * immune_rate_A
        A2S = A * gamma * (1 - immune_rate_A)

        dS = - S * transm_per_susc + R / dimmuneI + RA / dimmuneA - S * daily_vacc_per_elig + vacc_immun_loss + I2S + Ih2S + A2S  # susceptible & not vaccine-immune
        dE = - E / alpha + S * transm_per_susc + Vxd * transm_per_susc * delta_share  # exposed
        dI = (E * pS) / alpha - I * gamma  # infectious & symptomatic
        dIh = I * hosp * gamma - Ih / hlos  # hospitalized (not considered infectious)
        dA = E * (1 - pS) / alpha - A * gamma  # infectious asymptomatic
        dR = I2R + Ih2R - R / dimmuneI - R * daily_vacc_per_elig  # recovered from symp-not-hosp & immune & not vaccine-immune
        dRA = A2RA - RA / dimmuneA - RA * daily_vacc_per_elig  # recovered from asymptomatic & immune & not vaccine-immune
        dV = (S + R + RA) * daily_vacc_per_elig * (1 - delta_vacc_escape) - vacc_immun_loss * (1 - delta_vacc_escape)  # vaccine-immune
        dVxd = (S + R + RA) * daily_vacc_per_elig * delta_vacc_escape - vacc_immun_loss * delta_vacc_escape - Vxd * transm_per_susc * delta_share  # vaccine-immune except against delta (problem with too many people exiting)
        dD = dnh * I * gamma + dh * Ih / hlos  # death

        return dS, dE, dI, dIh, dA, dR, dRA, dV, dVxd, dD


    # the differential equation, takes y, outputs dy
    def seir(self, t, y):
        ydf = CovidModel.y_to_df(y)

        # get param and ef values from lookup table
        t_int = min(math.floor(t), len(self.trange) - 1)
        params = self.gparams_lookup[t_int]
        ef = self.ef_by_t[t_int]

        # build dy
        dy = []
        for group in self.groups:
            transm = CovidModel.daily_transmission_per_susc(ef, I_total=ydf['I'].sum(), A_total=ydf['A'].sum(), **params[group])
            dy += CovidModel.single_group_seir(
                y=y,
                single_group_y=list(ydf.loc[group, :]),
                transm_per_susc=transm,
                delta_share=self.variant_prevalence_df.loc[(t_int, group, 'delta'), 'e_prev'],
                **params[group])
                # **{**params[group], 'dnh': params[group]['dnh'] * max((1 + 0.001 * (ydf['Ih'].sum() - 1250)), 1)})

        return dy

    # the initial values for y
    def y0(self):
        y = []
        for group in self.groups:
            # the first initial value for each group (presumably uninfected) is the population, which we get from gparams
            y.append(self.gparams['groupN'][group] - (1 if group == 'group1' else 0))
            # everything else is 0, except...
            y += [0] * (len(self.vars) - 1)
        # ...we start with one infection in the first group
        y[2] = 2
        # return y
        return np.reshape(y, (len(self.groups), len(self.vars))).transpose().flatten()

    # solve the diff eq using scipy.integrate.solve_ivp; put the solution in    self.solution_y (list) and self.solution_ydf (dataframe)
    def solve_seir(self, seir=None):
        self.solution = spi.solve_ivp(fun=self.seir if seir is None else seir, t_span=[self.tmin, self.tmax], y0=self.y0(), t_eval=range(self.tmin, self.tmax))
        if not self.solution.success:
            raise RuntimeError(f'ODE solver failed with message: {self.solution.message}')
        self.solution_y = np.transpose(self.solution.y)
        self.solution_ydf_full = pd.concat([self.y_to_df(self.solution_y[t]) for t in self.trange], keys=self.trange, names=['t', 'group'])

    # count the total hosps by t as the sum of Ih and Ic
    def total_hosps(self):
        return self.solution_ydf_summed['Ih']

    # count the new exposed individuals by day
    def new_exposed(self):
        sum_df = self.solution_ydf_summed
        return sum_df['E'] - sum_df['E'].shift(1) + sum_df['E'].shift(1) / self.gparams['alpha']

    # create a new fit and assign to this model
    def gen_fit(self, engine, tags=None):
        fit = CovidModelFit(tslices=self.tslices, fixed_efs=self.efs, tags=tags)
        fit.fit_params = None
        fit.model = self
        self.fit_id = fit.write_to_db(engine)

    # write to stage.covid_model_results in Postgres
    def write_to_db(self, engine=None, tags=None, new_fit=False):
        if engine is None:
            engine = self.engine

        # if there's no existing fit assigned, create a new fit and assign that one
        if new_fit or self.fit_id is None:
            self.gen_fit(engine, tags)

        # get the summed solution, add null index for the group, and then append to group solutions
        summed = self.solution_ydf_summed
        summed['group'] = None
        summed.set_index('group', append=True, inplace=True)
        df = pd.concat([self.solution_ydf, summed])

        # join the ef values onto the dataframe
        ef_series = pd.Series(self.ef_by_t, name='ef').rename_axis('t')
        oef_series = pd.Series(self.obs_ef_by_t, name='observed_ef').rename_axis('t')
        df = df.join(ef_series).join(oef_series)

        # add estimated vaccine prevalence in each compartment
        for v in ['s', 'e', 'i', 'h', 'd']:
            df[f'vacc_prev_{v}'] = self.vacc_prevalence_df[v]

        # add fit_id and created_at date
        df['fit_id'] = self.fit_id
        df['created_at'] = dt.datetime.now()

        # add params
        gparams_df = pd.DataFrame(self.gparams_lookup).unstack().rename_axis(index=['t', 'group']).rename("params").map(lambda d: json.dumps(d, ensure_ascii=False))
        df = df.join(gparams_df, how='left')
        print(df['params'])

        # write to database
        df.to_sql('covid_model_results'
                  , con=engine, schema='stage'
                  , index=True, if_exists='append', method='multi')

    def write_gparams_lookup_to_csv(self, fname):
        df_by_t = {t: pd.DataFrame.from_dict(df_by_group, orient='index') for t, df_by_group in self.gparams_lookup.items()}
        pd.concat(df_by_t, names=['t', 'group']).to_csv(fname)


# class to find an optimal fit of transmission control (ef) values to produce model results that align with acutal hospitalizations
class CovidModelFit:

    def __init__(self, tslices, fixed_efs, fitted_efs=None, efs_cov=None, fit_params=None, actual_hosp=None, tags=None, model_params=None):
        self.tslices = tslices
        self.fixed_efs = fixed_efs
        self.fitted_efs = fitted_efs
        self.fitted_efs_cov = efs_cov
        self.tags = tags
        self.actual_hosp = actual_hosp
        self.model_params = model_params

        self.model = None

        self.fit_params = {} if fit_params is None else fit_params

        # set fit param efs0
        if 'efs0' not in self.fit_params.keys():
            self.fit_params['efs0'] = [0.75] * self.fit_count
        elif type(self.fit_params['efs0']) in (float, int):
            self.fit_params['efs0'] = [self.fit_params['efs0']] * self.fit_count

        # set fit params ef_min and ef_max
        if 'ef_min' not in self.fit_params.keys():
            self.fit_params['ef_min'] = 0.00
        if 'ef_max' not in self.fit_params.keys():
            self.fit_params['ef_max'] = 0.99

    @staticmethod
    def from_db(conn, fit_id):
        df = pd.read_sql_query(f"select * from stage.covid_model_fits where id = '{fit_id}'", con=conn, coerce_float=True)
        fit_count = len(df['efs_cov'][0]) if df['efs_cov'][0] is not None else df['fit_params'][0]['fit_count']
        return CovidModelFit(
            tslices=df['tslices'][0]
            , fixed_efs=df['efs'][0][:-fit_count]
            , fitted_efs=df['efs'][0][-fit_count:]
            , efs_cov=df['efs_cov'][0]
            , fit_params=df['fit_params'][0]
            , tags=df['tags'][0]
            , model_params=df['model_params'][0])

    # the number of total ef values, including fixed values
    @property
    def ef_count(self):
        return len(self.tslices) - 1

    # the number of variables to be fit
    @property
    def fit_count(self):
        return self.ef_count - len(self.fixed_efs)

    @property
    def efs(self):
        return (list(self.fixed_efs) if self.fixed_efs is not None else []) + (list(self.fitted_efs) if self.fitted_efs is not None else [])

    # add a tag, to make fits easier to query
    def add_tag(self, tag_type, tag_value):
        self.tags[tag_type] = tag_value

    # runs the model using a given set of efs, and returns the modeled hosp values (which will be fit to actual hosp data)
    def run_model_and_get_total_hosps(self, ef_by_slice):
        extended_ef_by_slice = self.fixed_efs + list(ef_by_slice)
        self.model.apply_tc(extended_ef_by_slice)
        self.model.solve_seir()
        return self.model.total_hosps()

    # the cost function: runs the model using a given set of efs, and returns the sum of the squared residuals
    def cost(self, ef_by_slice: list):
        modeled = self.run_model_and_get_total_hosps(ef_by_slice)
        res = [m - a if not np.isnan(a) else 0 for m, a in zip(modeled, list(self.actual_hosp))]
        c = sum(e**2 for e in res)
        return c

    # the cost function to be used for the particle swarm
    def ps_cost(self, xs):
        return sum(self.cost(x) for x in xs)

    # run an optimization to minimize the cost function using scipy.optimize.minimize()
    # method = 'curve_fit' or 'minimize'
    def run(self, engine, method='curve_fit', **model_params):

        if self.actual_hosp is None:
            self.actual_hosp = ExternalHosps(engine, self.model.start_date).fetch('emresource_hosps.csv')

        self.model = CovidModel(tslices=self.tslices, efs=self.fixed_efs + self.fit_params['efs0'], engine=engine)
        self.model.prep(**model_params)

        # run fit
        if self.model.efs != self.fixed_efs:
            if method == 'curve_fit':
                def func(trange, *efs):
                    return self.run_model_and_get_total_hosps(efs)
                self.fitted_efs, self.fitted_efs_cov = spo.curve_fit(
                    f=func
                    , xdata=self.model.trange
                    , ydata=self.actual_hosp[:len(self.model.trange)]
                    , p0=self.fit_params['efs0']
                    , bounds=([self.fit_params['ef_min']] * self.fit_count, [self.fit_params['ef_max']] * self.fit_count))
            elif method == 'minimize':
                minimization_results = spo.minimize(
                    lambda x: self.cost(x)
                    , self.fit_params['efs0']
                    , method='L-BFGS-B'
                    , bounds=[(self.fit_params['ef_min'], self.fit_params['ef_max'])] * self.fit_count
                    , options=self.fit_params)
                print(minimization_results)
                self.fitted_efs = minimization_results.x
            elif method == 'pswarm':
                options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
                bounds = ([self.fit_params['ef_min']] * self.fit_count, [self.fit_params['ef_max']] * self.fit_count)
                optimizer = ps.single.GlobalBestPSO(n_particles=self.fit_count * 2, dimensions=self.fit_count, bounds=bounds, options=options)
                cost, self.fitted_efs = optimizer.optimize(self.ps_cost, iters=10)
                print(self.fitted_efs)

            self.model.set_ef_by_t(self.efs)

    # write the fit as a single row to stage.covid_model_fits, describing the optimized ef values
    def write_to_db(self, engine):
        metadata = MetaData(schema='stage')
        metadata.reflect(engine, only=['covid_model_fits'])
        fits_table = metadata.tables['stage.covid_model_fits']

        if self.fit_params is not None:
            self.fit_params['fit_count'] = self.fit_count

        stmt = fits_table.insert().values(tslices=[int(x) for x in self.tslices],
                                          model_params=self.model.raw_params if self.model is not None else None,
                                          fit_params=self.fit_params,
                                          efs=list(self.efs),
                                          observed_efs=self.model.obs_ef_by_slice if self.model is not None else None,
                                          created_at=datetime.now(),
                                          tags=self.tags,
                                          efs_cov=[list(a) for a in self.fitted_efs_cov] if self.fitted_efs_cov is not None else None)

        conn = engine.connect()
        result = conn.execute(stmt)

        if self.model is not None:
            self.model.fit_id = result.inserted_primary_key[0]

        return result.inserted_primary_key[0]

