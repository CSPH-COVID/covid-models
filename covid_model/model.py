### Python Standard Library ###
import json
import math
import datetime as dt
import copy
from operator import itemgetter
from collections import OrderedDict, defaultdict
### Third Party Imports ###
import numpy as np
import pandas as pd
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import scipy.integrate as spi
import scipy.sparse as spsp
### Local Imports ###
from ode_flow_terms import ConstantODEFlowTerm, ODEFlowTerm
from data_imports import ExternalVacc, get_region_mobility_from_db
from covid_model.utils import get_params


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel:
    ####################################################################################################################
    ### Setup

    # basic model info
    attrs = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                        'priorinf': ['none', 'non-omicron', 'omicron'],
                        'variant': ['none', 'alpha', 'delta', 'omicron', 'ba2'],
                        'immun': ['none', 'weak', 'strong'],
                        'region': ['co']})

    param_attr_names = ('age', 'vacc', 'priorinf', 'variant', 'immun', 'region')
    tags = {}
    __tmin = 0

    solution = None
    solution_y = None
    solution_ydf = None
    tslices = None
    tc = None
    tc_cov = None

    # model settings
    vacc_proj_params = None
    timeseries_effects = {}
    model_region_definitions = None
    model_mobility_mode = None
    actual_mobility = {}
    mobility_proj_params = None
    actual_vacc_df = None
    proj_vacc_df = None
    _params_defs = None
    _region_defs = None
    _attribute_multipliers = None
    _mab_prevalence = None
    params_by_t = None
    spec_ids = None
    region_fit_spec_ids = None
    region_fit_result_ids = None

    # ode settings
    t_prev_lookup = None
    t_next_lookup = None
    terms = None
    linear_matrix = None
    nonlinear_matrices = None
    constant_vector = None
    nonlinear_multiplier = None
    recently_updated_properties = []


    def __init__(self, engine=None, base_model=None, update_derived_properties=True, **margs):

        # if there is a base model, take all its attributes
        if base_model is not None:
            for key, val in base_model.__dict__():
                setattr(self, key, copy.deepcopy(val))

        # update any attributes with items in margs
        for key, val in margs:
            setattr(self, key, copy.deepcopy(val))
            self.recently_updated_properties.append(key)

        if update_derived_properties:
            self.update(engine)

    # some properties are derived / computed / constructed. If the non-derived properties are updated, the derived
    # properties may need recomputing
    def update(self, engine):
        if 'end_date' in self.recently_updated_properties:
            self.set_actual_vacc(engine, self.actual_vacc_df)

        if any([p in self.recently_updated_properties for p in ['end_date', 'vacc_proj_params']]):
            self.set_proj_vacc()

        if any([p in self.recently_updated_properties for p in ['end_date', 'mab_prevalence', 'timeseries_effect_multipliers']]):
            self.add_timeseries_effect('mab', self.mab_prevalence, self.timeseries_effect_multipliers, fill_forward=True)

        if 'end_date' in self.recently_updated_properties:
            self.set_actual_mobility(engine)

        if any([p in self.recently_updated_properties for p in ['end_date', 'mobility_proj_params']]):
            self.set_proj_mobility()

        if any([p in self.recently_updated_properties for p in ['end_date', 'model_mobility_mode', 'mobility_proj_params']]):
            if self.model_mobility_mode != "none":
                self.params_defs.update(self.get_mobility_as_params())

        if any([p in self.recently_updated_properties for p in ['region_fit_spec_ids', 'region_fit_result_ids']]):
            self.params_defs.update(self.get_kappas_as_params_using_region_fits(engine))

        self.recently_updated_properties = []


    ### FUNCTIONS TO UPDATE DERIVED PROPERTIES ###

    def set_actual_vacc(self, engine, actual_vacc_df=None):
        if engine is not None:
            actual_vacc_df_list = []
            for region in self.regions:
                county_ids = self.model_region_definitions[region]['counties_fips']
                actual_vacc_df_list.append(ExternalVacc(engine, t0_date=self.start_date).fetch(county_ids=county_ids).assign(region=region).set_index('region', append=True).reorder_levels(['t', 'region', 'age']))
            self.actual_vacc_df = pd.concat(actual_vacc_df_list)
        if actual_vacc_df is not None:
            self.actual_vacc_df = actual_vacc_df.copy()

    def set_proj_vacc(self):
        proj_lookback = self.vacc_proj_params['lookback']
        proj_fixed_rates = self.vacc_proj_params['fixed_rates']
        max_cumu = self.vacc_proj_params['max_cumu']
        max_rate_per_remaining = self.vacc_proj_params['max_rate_per_remaining']
        realloc_priority = self.vacc_proj_params['realloc_priority']

        shots = list(self.actual_vacc_df.columns)
        region_df = pd.DataFrame({'region': self.regions})

        # add projections
        proj_from_t = self.actual_vacc_df.index.get_level_values('t').max() + 1
        proj_to_t = (self.end_date - self.start_date).days + 1
        if proj_to_t >= proj_from_t:
            proj_trange = range(proj_from_t, proj_to_t)
            # project rates based on the last {proj_lookback} days of data
            projected_rates = self.actual_vacc_df[self.actual_vacc_df.index.get_level_values(0) >= proj_from_t-proj_lookback].groupby(['region', 'age']).sum()/proj_lookback
            # override rates using fixed values from proj_fixed_rates, when present
            if proj_fixed_rates:
                proj_fixed_rates_df = pd.DataFrame(proj_fixed_rates).rename_axis(index='age').reset_index().merge(region_df,how='cross').set_index(['region', 'age'])
                for shot in shots:
                    # TODO: currently treats all regions the same. Need to change if finer control desired
                    projected_rates[shot] = proj_fixed_rates_df[shot]
            # build projections
            projections = pd.concat({t: projected_rates for t in proj_trange}).rename_axis(index=['t', 'region', 'age'])

            # reduce rates to prevent cumulative vaccination from exceeding max_cumu
            if max_cumu:
                cumu_vacc = self.actual_vacc_df.groupby(['region', 'age']).sum()
                groups = realloc_priority if realloc_priority else projections.groupby(['region','age']).sum().index
                populations = [{'age': li['attributes']['age'], 'region': li['attributes']['region'], 'population': li['values']} for li in self.params_defs['group_pop'] if li['attributes']['region'] in self.regions]
                for t in projections.index.unique('t'):
                    this_max_cumu = get_params(max_cumu.copy(), t)

                    # TODO: currently treats all regions the same. Need to change if finer control desired
                    max_cumu_df = pd.DataFrame(this_max_cumu).rename_axis(index='age').reset_index().merge(region_df, how='cross').set_index(['region', 'age']).sort_index()
                    max_cumu_df = max_cumu_df.mul(pd.DataFrame(populations).set_index(['region', 'age'])['population'], axis=0)
                    for i in range(len(groups)):
                        group = groups[i]
                        key = tuple([t] + list(group))
                        current_rate = projections.loc[key]
                        max_rate = max_rate_per_remaining * (max_cumu_df.loc[group] - cumu_vacc.loc[group])
                        excess_rate = (projections.loc[key] - max_rate).clip(lower=0)
                        projections.loc[key] -= excess_rate
                        # if a reallocate_order is provided, reallocate excess rate to other groups
                        if i < len(groups) - 1 and realloc_priority is not None:
                            projections.loc[tuple([t] + list(groups[i + 1]))] += excess_rate

                    cumu_vacc += projections.loc[t]

            self.proj_vacc_df = projections
        else:
            self.proj_vacc_df = None

    def set_actual_mobility(self, engine):
        regions = self.regions
        county_ids = [fips for region in regions for fips in self.model_region_definitions[region]['counties_fips']]
        df = get_region_mobility_from_db(engine, county_ids=county_ids).reset_index('measure_date')

        # add regions to dataframe
        regions_lookup = {fips: region for region in regions for fips in self.model_region_definitions[region]['counties_fips']}
        df['origin_region'] = [regions_lookup[id] for id in df['origin_county_id']]
        df['destination_region'] = [regions_lookup[id] for id in df['destination_county_id']]
        df['t'] = (df['measure_date'] - self.start_date).dt.days
        # find most recent data before self.start_date and set its time to zero so we have an initial mobility
        df.replace({'t': max(df['t'][df['t']<=0])}, 0, inplace=True)

        df = df[df['t']>=0].drop(columns=['origin_county_id', 'destination_county_id']) \
            .groupby(['t', 'origin_region', 'destination_region']) \
            .aggregate(total_dwell_duration_hrs=('total_dwell_duration_hrs', 'sum'))

        # Create dictionaries of matrices, both D and M.
        ts = df.index.get_level_values('t')
        region_idx = {region: i for i, region in enumerate(regions)}
        dwell_matrices = {}
        for t in ts:
            dfsub = df.loc[df.index.get_level_values('t') == t].reset_index('t', drop=True).reset_index()
            idx_i = [region_idx[region] for region in dfsub['origin_region']]
            idx_j = [region_idx[region] for region in dfsub['destination_region']]
            vals = dfsub['total_dwell_duration_hrs']
            dwell = spsp.coo_array((vals, (idx_i, idx_j)), shape=(len(regions), len(regions))).todense()
            dwell[np.isnan(dwell)] = 0
            dwell_rownorm = dwell / dwell.sum(axis=1)[:, np.newaxis]
            dwell_colnorm = dwell / dwell.sum(axis=0)[np.newaxis, :]
            dwell_matrices[t] = {"dwell": dwell.tolist(), "dwell_rownorm": dwell_rownorm.tolist(), "dwell_colnorm": dwell_colnorm.tolist()}

        self.actual_mobility = dwell_matrices

    def set_proj_mobility(self):
        # TODO: implement mobility projections
        self.proj_mobility = {}

    def add_timeseries_effect(self, effect_type_name, prevalence_df, multiplier_dict, fill_forward=False):
        # build prevalence and multiplier dataframes from inputs
        prevalence_df = prevalence_df[prevalence_df.max(axis=1) > 0]
        if fill_forward and self.end_date > prevalence_df.index.max().date():
            projections = pd.DataFrame.from_dict({date: prevalence_df.iloc[-1] for date in pd.date_range(prevalence_df.index.max() + dt.timedelta(days=1), self.end_date)}, orient='index')
            prevalence_df = pd.concat([prevalence_df, projections]).sort_index()

        self.timeseries_effects[effect_type_name] = []
        for effect_name in prevalence_df.columns:
            d = {'effect_name': effect_name, 'multipliers': multiplier_dict[effect_name], 'start_date': prevalence_df.index.min().strftime('%Y-%m-%d'), 'prevalence': list(prevalence_df[effect_name].values)}
            self.timeseries_effects[effect_type_name].append(d)

    def get_mobility_as_params(self):
        mobility_dict = self.actual_mobility
        mobility_dict.update(self.proj_mobility)
        tslices = list(mobility_dict.keys())
        params = {}
        if self.model_mobility_mode == "population_attached":
            matrix_list = [np.dot(mobility_dict[t]['dwell_rownorm'], np.transpose(mobility_dict[t]['dwell_colnorm'])) for t in tslices]
            for j, from_region in enumerate(self.regions):
                params[f"mob_{from_region}"] = [{'tslices': tslices[1:], 'attributes': {'region': to_region}, 'values': [m[i,j] for m in matrix_list]} for i, to_region in enumerate(self.regions)]
        elif self.model_mobility_mode == "location_attached":
            dwell_rownorm_list = [mobility_dict[t]['dwell_rownorm'] for t in tslices]
            dwell_colnorm_list = [mobility_dict[t]['dwell_colnorm'] for t in tslices]
            for j, in_region in enumerate(self.regions):
                params[f"mob_fracin_{in_region}"] = [{'tslices': tslices[1:], 'attributes': {'seir': 'S', 'region': to_region}, 'values': [m[i,j] for m in dwell_rownorm_list]} for i, to_region in enumerate(self.regions)]
            for i, from_region in enumerate(self.regions):
                for j, in_region in enumerate(self.regions):
                    params[f"mob_{in_region}_fracfrom_{from_region}"] = [{'tslices': tslices[1:], 'attributes': {},  'values': [m[i,j] for m in dwell_colnorm_list]}]
        else:
            raise ValueError(f'Mobility mode {self.model_mobility_mode} not supported')
        # add in region populations as parameters for use later
        region_pops = {params_list['attributes']['region']: params_list['values'] for params_list in self.params_defs['total_pop'] }
        params.update(
            {f"region_pop_{region}": [{'tslices': None, 'attributes': {}, 'values': region_pops[region] }] for region in self.regions}
        )
        return params

    def get_kappas_as_params_using_region_fits(self, engine):
        # TODO: load number of infected from database to compute mobility aware kappas
        # will set TC = 0.0 for the model and scale the "kappa" parameter for each region according to its fitted TC values so the forward sim can be run.
        # This approach does not currently support fitting, since only TC can be fit and there's only one set of TC for the model.
        results_list = []
        for i, tup in enumerate(zip(self.regions, self.region_fit_spec_ids)):
            region, spec_id = tup
            # load tslices and tcs from the database
            df = pd.read_sql_query(f"select regions, start_date, end_date, tslices, tc, from covid_model.specifications where spec_id = {spec_id}", con=engine, coerce_float=True)
            # make sure region in run spec matches our region
            if json.loads(df['regions'][0])[0] != region:
                ValueError(f'spec_id {spec_id} has region {json.loads(df["regions"][0])[0]} which does not match model\'s {i}th region: {region}')
            tslices = [(df['start_date'][0] + dt.timedelta(days=d).days - self.start_date).days for d in [0] + df['tslices'][0]]
            tc = df['tc'][0]
            results_list.append(pd.DataFrame.from_dict({'tslices': tslices, region: tc}).set_index('tslices'))
        # union all the region tslices
        df_tcs = pd.concat(results_list, axis=1)
        tslices = df_tcs.index.drop([0]).to_list()

        # retrieve prevalence data from db if allowing mobility
        prev_df = None
        if self.model_mobility_mode != 'none':
            results_list = []
            for region, spec_id, result_id in zip(self.regions, self.region_fit_spec_ids, self.region_fit_result_ids):
                df = pd.read_sql_query(f"SELECT t, vals FROM covid_model.results_v2 WHERE spec_id = {spec_id} AND result_id = {result_id} order by t", con=engine, coerce_float=True)
                df['infected'] = [sum(itemgetter('I', 'Ih')(json.loads(row))) for row in df['vals']]
                df['pop'] = [sum(json.loads(row).values()) for row in df['vals']]
                df = df.groupby('t').sum(['infected', 'pop'])
                df[region]=df['infected']/df['pop']
                df = df.drop(columns=['infected', 'pop'])
                results_list.append(df)
            prev_df = pd.concat(results_list, axis=1)

        # compute kappa parameter for each region and apply to model parameters
        params = {}
        if self.model_mobility_mode == 'none':
            self.tslices = tslices
            self.tc = [0.0] * (len(tslices) + 1)
            params = {'kappa': [{'tslices': tslices, 'attributes': {'region': region}, 'values': [(1-tc) for tc in df_tcs[region]]} for region in df_tcs.columns]}
        elif self.model_mobility_mode == 'location_attached':
            # TODO: Implement
            params = {}
        elif self.model_mobility_mode == 'population_attached':
            # update tslices to include any t where mobility changes
            mob_tslices = np.array([t for t in self.actual_mobility.keys() if t >= self.tmin and t < self.tmax])
            prev_tslices = np.array([t for t in prev_df.index if t >= self.tmin and t < self.tmax])

            combined_tslices = [t.item() if isinstance(t, np.int32) else t for t in sorted(list(set([0] + tslices).union(set(mob_tslices)).union(set(prev_tslices))))]
            kappas = np.zeros(shape=[len(combined_tslices), len(self.regions)])
            for i, t in enumerate(combined_tslices):
                tc = df_tcs.iloc[df_tcs.index <= t,].iloc[-1].to_numpy()
                prev = prev_df.iloc[prev_tslices[prev_tslices <= t][-1]].to_numpy()
                mobs = self.actual_mobility[mob_tslices[mob_tslices <= t][-1]]
                kappas[i,] = (1-tc) * prev / np.linalg.multi_dot([mobs['dwell_rownorm'], np.transpose(mobs['dwell_colnorm']), np.transpose(prev)])
            np.nan_to_num(kappas, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.tslices = combined_tslices[1:]
            self.tc = [0.0] * len(combined_tslices)
            params = {'kappa_pa': [{'tslices': combined_tslices[1:], 'attributes': {'region': region}, 'values': kappas[:,j].tolist()} for j, region in enumerate(self.regions)]}
        return params


    ####################################################################################################################
    ### Properites

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, value):
        self._start_date = value if isinstance(value, dt.datetime.date) else dt.datetime.strptime(value, "%Y-%m-%d").date()

    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, value):
        self._end_date = value if isinstance(value, dt.datetime.date) else dt.datetime.strptime(value, "%Y-%m-%d").date()

    @property
    def regions(self):
        return self._regions

    @regions.setter
    def regions(self, value: list):
        self._regions = value
        self.attrs['regions'] = value  # if regions changes, update the compartment attributes also

    @property
    def params_defs(self):
        return self._params_defs

    @params_defs.setter
    def params_defs(self, value):
        self._params_defs = value if isinstance(value, dict) else json.load(open(value))

    @property
    def region_defs(self):
        return self._region_defs

    @region_defs.setter
    def region_defs(self, value):
        self._region_defs = value if isinstance(value, dict) else json.load(open(value))

    @property
    def attribute_multipliers(self):
        return self._attribute_multipliers

    @attribute_multipliers.setter
    def attribute_multipliers(self, value):
        self._attribute_multipliers = value if isinstance(value, list) else json.load(open(value))

    @property
    def mab_prevalence(self):
        return self._mab_prevalence

    @mab_prevalence.setter
    def mab_prevalence(self, value):
        self._mab_prevalence = pd.read_csv(value, parse_dates=['date'], index_col=0) if isinstance(value, str) else value

    @property
    def timeseries_effect_multipliers(self):
        return self._timeseries_effect_multipliers

    @timeseries_effect_multipliers.setter
    def timeseries_effect_multipliers(self, value):
        self._timeseries_effect_multipliers = value if isinstance(value, list) else json.load(open(value))


    # Properties that take a little computation to get

    @property
    def tmin(self):
        return self.__tmin

    @property
    def tmax(self):
        return (self.end_date - self.start_date).days + 1

    @property
    def trange(self):
        return range(self.tmin, self.tmax + 1)

    @property
    def daterange(self):
        return pd.date_range(self.start_date, end=self.end_date - dt.timedelta(days=1))

    @property
    def ndays(self):
        return (self.end_date - self.start_date).days

    @property
    def tslices_dates(self):
        return [self.start_date + dt.timedelta(days=ts) for ts in [0] + self.tslices]



    # initial state y0, expressed as a dictionary with non-empty compartments as keys
    @property
    def y0_dict(self):
        group_pops = {(li['attributes']['region'], li['attributes']['age']): li['values'] for li in
                      self.params_defs['group_pop'] if li['attributes']['region'] in self.attrs['region']}
        y0d = {('S', age, 'none', 'none', 'none', 'none', region): n for (region, age), n in group_pops.items()}
        return y0d

    # return the parameters as a dataframe with t and compartments as index and parameters as columns
    @property
    def params_as_df(self):
        return pd.concat({t: pd.DataFrame.from_dict(self.params_by_t[self.t_prev_lookup[t]], orient='index') for t in
                          self.trange}).rename_axis(index=['t'] + list(self.param_attr_names))




    ####################################################################################################################
    ### useful getters

    def get_vacc_rates(self):
        df = pd.concat([self.actual_vacc_df, self.proj_vacc_df])
        return df

    # create a y0 vector with all values as 0, except those designated in y0_dict
    def y0_from_dict(self, y0_dict):
        y0 = [0] * self.length
        for cmpt, n in y0_dict.items():
            y0[self.cmpt_idx_lookup[cmpt]] = n
        return y0


    def get_all_county_fips(self, regions=None):
        return [county_fips for region in regions for county_fips in self.model_region_definitions[region]['counties_fips']]


    # convert y-array to series with compartment attributes as multiindex
    def y_to_series(self, y):
        return pd.Series(index=self.compartments_as_index, data=y)

    # return solution grouped by group_by_attr_levels
    def solution_sum(self, group_by_attr_levels):
        return self.solution_ydf.groupby(group_by_attr_levels, axis=1).sum()

    def ode_terms_as_json(self, compact=False):
        if compact:
            cm = ", ".join([f'[{i},{c}]' for i, c in enumerate(self.compartments)])
            cv = [[t, spsp.csr_array(vec)] for t, vec in self.constant_vector.items() if any(vec != 0)]
            cv = {t: ' ,'.join([f'({idx},{val:.2e})' for idx, val in zip(m.nonzero()[1].tolist(), m[m.nonzero()].tolist())]) for t, m in cv}
            lm = {t: ' ,'.join([f'({idx1},{idx2},{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[0].tolist(), m.nonzero()[1].tolist(), m[m.nonzero()].A[0].tolist())]) for t, m in self.linear_matrix.items() if len(m.nonzero()[0]) > 0}
            nl = {t: {f'({",".join([f"{k}" for k in keys])})': ', '.join([f'({idx1},{idx2},{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[0].tolist(), m.nonzero()[1].tolist(), m[m.nonzero()].A[0].tolist()) if val != 0]) for keys, m in mat_dict.items()} for t, mat_dict in self.nonlinear_matrices.items() if len(mat_dict) > 0}
            nlm = self.nonlinear_multiplier
            return json.dumps({"compartments": cm, "constant_vector": cv, "linear_matrix": lm, "nonlinear_multiplier": nlm, "nonlinear_matrices": nl}, indent=2)
        else:
            def fcm(i):
                return f'{",".join(self.compartments[i])}'

            cv = [[t, spsp.csr_array(vec)] for t, vec in self.constant_vector.items() if any(vec != 0)]
            cv = {t: {fcm(idx): f'{val:.2e}' for idx, val in zip(m.nonzero()[1].tolist(), m[m.nonzero()].tolist())} for t, m in cv}
            lm = {t: {f'({fcm(idx1)};{fcm(idx2)}': f'{val:.2e}' for idx1, idx2, val in zip(m.nonzero()[1].tolist(), m.nonzero()[0].tolist(), m[m.nonzero()].A[0].tolist())} for t, m in self.linear_matrix.items() if len(m.nonzero()[0]) > 0}
            nl = {t: {f'({";".join([f"{fcm(k)}" for k in keys])})': {f'({fcm(idx1)};{fcm(idx2)})': f'{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[1].tolist(), m.nonzero()[0].tolist(), m[m.nonzero()].A[ 0].tolist()) if val != 0} for keys, m in mat_dict.items()} for t, mat_dict in self.nonlinear_matrices.items() if len(mat_dict) > 0}
            nlm = self.nonlinear_multiplier
            return json.dumps({"constant_vector": cv, "linear_matrix": lm, "nonlinear_multiplier": nlm, "nonlinear_matrices": nl}, indent=2)


    # immunity
    def immunity(self, variant='omicron', vacc_only=False, to_hosp=False, age=None):
        params = self.params_as_df
        group_by_attr_names = [attr_name for attr_name in self.param_attr_names if attr_name != 'variant']
        n = self.solution_sum(group_by_attr_names).stack(level=group_by_attr_names)

        if age is not None:
            params = params.xs(age, level='age')
            n = n.xs(age, level='age')

        if vacc_only:
            params.loc[params.index.get_level_values('vacc') == 'none', 'immunity'] = 0
            params.loc[params.index.get_level_values('vacc') == 'none', 'severe_immunity'] = 0

        variant_params = params.xs(variant, level='variant')
        if to_hosp:
            weights = variant_params['hosp'] * n
            return (weights * (1 - (1 - variant_params['immunity']) * (1 - variant_params['severe_immunity']))).groupby('t').sum() / weights.groupby('t').sum()
        else:
            return (n * variant_params['immunity']).groupby('t').sum() / n.groupby('t').sum()


    ### ODE RELATED ###

    # get the level associated with a given attribute name
    # e.g. if attributes are ['seir', 'age', 'variant'], the level of 'age' is 1 and the level of 'variant' is 2
    def attr_level(self, attr_name):
        return list(self.attrs.keys()).index(attr_name)

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
        return [cmpt for cmpt in (self.param_compartments if is_param_cmpts else self.compartments) if
                self.does_cmpt_have_attrs(cmpt, attrs, is_param_cmpts)]

    # return the "first" compartment that matches a dictionary of attributes, with "first" determined by attribute order
    def get_default_cmpt_by_attrs(self, attrs):
        return tuple(attrs[attr_name] if attr_name in attrs.keys() else attr_list[0] for attr_name, attr_list in self.attrs.items())

    # get a parameter for a given set of attributes and trange
    def get_param(self, param, attrs=None, trange=None):
        actual_trange = self.trange if trange is None else set(self.trange).intersection(trange)
        cmpt_list = self.filter_cmpts_by_attrs(attrs, is_param_cmpts=True) if attrs else self.param_compartments
        return [(cmpt, [self.params_by_t[t][cmpt][param] for t in actual_trange]) for cmpt in cmpt_list]

    # get all terms that refer to flow from one specific compartment to another
    def get_terms_by_cmpt(self, from_cmpt, to_cmpt):
        return [term for term in self.terms if
                term.from_cmpt_idx == self.cmpt_idx_lookup[from_cmpt] and term.to_cmpt_idx == self.cmpt_idx_lookup[
                    to_cmpt]]

    # get the indices of all terms that refer to flow from one specific compartment to another
    def get_term_indices_by_attr(self, from_attrs, to_attrs):
        return [i for i, term in enumerate(self.terms) if
                self.does_cmpt_have_attrs(self.compartments[term.from_cmpt_idx],
                                          from_attrs) and self.does_cmpt_have_attrs(
                    self.compartments[term.to_cmpt_idx], to_attrs)]

    # get the terms that refer to flow from compartments with a set of attributes to compartments with another set of attributes
    def get_terms_by_attr(self, from_attrs, to_attrs):
        return [self.terms[i] for i in self.get_term_indices_by_attr(from_attrs, to_attrs)]


    ####################################################################################################################
    ### Prepping and Running

    def get_timeseries_effect_multipliers(self):
        params = set().union(*[effect_specs['multipliers'].keys() for effects in self.timeseries_effects.values() for effect_specs in effects])
        multipliers = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date), columns=params, data=1.0)

        for effect_type in self.timeseries_effects.keys():
            multiplier_dict = {}
            prevalence_df = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date))

            for effect_specs in self.timeseries_effects[effect_type]:
                effect_start_date = dt.datetime.strptime(effect_specs['start_date'], '%Y-%m-%d').date()
                effect_end_date = effect_start_date + dt.timedelta(days=len(effect_specs['prevalence'])-1)
                effect_df = pd.DataFrame(index=pd.date_range(effect_start_date, effect_end_date))
                effect_df[effect_specs['effect_name']] = effect_specs['prevalence']
                # effect dates must overlap with at least one model date
                if len(prevalence_df.index.intersection(effect_df.index)) > 0:
                    # missing values at beginning are set to 0; missing dates at end are filled forward from last valid value
                    prevalence_df = prevalence_df.merge(effect_df, how="left", left_index=True, right_index=True).sort_index().ffill().fillna(0)
                    multiplier_dict[effect_specs['effect_name']] = {**{param: 1.0 for param in params}, **effect_specs['multipliers']}

            if len(multiplier_dict) > 0:
                prevalence_df = prevalence_df.sort_index()
                multiplier_df = pd.DataFrame.from_dict(multiplier_dict, orient='index').rename_axis(index='effect').fillna(1)

                prevalence = prevalence_df.stack().rename_axis(index=['t', 'effect'])
                remainder = 1 - prevalence.groupby('t').sum()

                multipliers_for_this_effect_type = multiplier_df.multiply(prevalence, axis=0).groupby('t').sum().add(remainder, axis=0)
                multipliers = multipliers.multiply(multipliers_for_this_effect_type)

        multipliers.index = (multipliers.index.to_series().dt.date - self.start_date).dt.days

        return multipliers

    def get_vacc_per_available(self):
        vacc_rates = self.get_vacc_rates()
        populations = [{'age': li['attributes']['age'], 'region': li['attributes']['region'], 'population':li['values']} for li in self.params_defs['group_pop'] if li['attributes']['region'] in self.regions]
        populations = pd.DataFrame(populations).set_index(['region','age'])
        cumu_vacc = vacc_rates.groupby(['region', 'age']).cumsum()
        cumu_vacc_final_shot = cumu_vacc - cumu_vacc.shift(-1, axis=1).fillna(0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.join(populations)
        # vaccinations eventually overtake population (data issue) which would make 'none' < 0 so clip at 0
        cumu_vacc_final_shot['none'] = (cumu_vacc_final_shot['population'] * 2 - cumu_vacc_final_shot.sum(axis=1)).clip(lower=0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.drop(columns='population')
        cumu_vacc_final_shot = cumu_vacc_final_shot.reindex(columns=['none', 'shot1', 'shot2', 'shot3'])

        available_for_vacc = cumu_vacc_final_shot.shift(1, axis=1).drop(columns='none')
        vacc_per_available = (vacc_rates / available_for_vacc).fillna(0).replace(np.inf, 0).reorder_levels(['t', 'region', 'age']).sort_index()
        # because vaccinations exceed the population, we can get rates greater than 1. To prevent compartments have negative people, we have to cap the rate at 1
        vacc_per_available = vacc_per_available.clip(upper=1)
        return vacc_per_available

    # set a single parameter (if val is provided), or apply a multiplier
    def set_param(self, param, val=None, attrs=None, trange=None, mult=None, except_attrs=None, desc=None):
        if val is not None:
            def apply(t, cmpt, param):
                self.params_by_t[t][cmpt][param] = val
        elif mult is not None:
            def apply(t, cmpt, param):
                self.params_by_t[t][cmpt][param] *= mult
        else:
            raise ValueError('Must provide val or mult')
        if type(val if val is not None else mult if mult is not None else val) not in (int, float, np.float64):
            raise TypeError(
                f'Parameter value (or multiplier) must be numeric; {val if val is not None else mult} is {type(val if val is not None else mult)}')
        if trange is None:
            actual_trange = self.params_trange
        else:
            actual_trange = set(self.params_trange).intersection(trange)
        cmpts = self.param_compartments
        if attrs:
            for attr in [attrs] if isinstance(attrs, dict) else attrs:
                cmpts = self.filter_cmpts_by_attrs(attr, is_param_cmpts=True)
        if except_attrs:
            for except_attr in [except_attrs] if isinstance(except_attrs, dict) else except_attrs:
                cmpts = [cmpt for cmpt in cmpts if
                         cmpt not in self.filter_cmpts_by_attrs(except_attr, is_param_cmpts=True)]
        for cmpt in cmpts:
            for t in actual_trange:
                apply(t, cmpt, param)

    # combine param_defs, vaccine_defs, attribute multipliers, etc. into a time indexed parameters dictionary
    def build_param_lookups(self, apply_vaccines=True, vacc_delay=14):
        # if increment is None, set trange to match TC tslices, with breaks added anywhere that has a tslice in model_params
        model_param_tslices = set()
        for param, param_specs_list in self.params_defs.items():
            for param_specs in param_specs_list:
                if 'tslices' in param_specs.keys() and param_specs['tslices'] is not None:
                    for tslice in param_specs['tslices']:
                        model_param_tslices.add((dt.datetime.strptime(tslice,
                                                                      "%Y-%m-%d").date() - self.start_date).days if isinstance(
                            tslice, str) else tslice)
        trange = sorted(list(set(self.tslices).union({0}).union({self.tmax}).union(model_param_tslices)))
        trange = [ts for ts in trange if ts < self.tmax and ts >= self.tmin]
        ## where should the above go?

        for param_name, param_list in self.params_defs.items():
            for param_dict in param_list:
                if param_dict['tslices']:
                    for i, (tmin, tmax) in enumerate(zip([self.tmin] + param_dict['tslices'], param_dict['tslices'] + [self.tmax])):
                        tmin = (dt.datetime.strptime(tmin, "%Y-%m-%d").date() - self.start_date).days if isinstance(tmin, str) else tmin
                        tmax = (dt.datetime.strptime(tmax, "%Y-%m-%d").date() - self.start_date).days if isinstance(tmax, str) else tmax
                        v = {a: av[i] for a, av in param_dict['values'].items()} if isinstance(param_dict['values'], dict) else param_dict['values'][i]
                        self.set_param(param_name, v, param_dict['attributes'], trange=range(tmin, tmax))
                else:
                    self.set_param(param_name, param_dict['values'], param_dict['attributes'])

        if apply_vaccines:
            vacc_per_available = self.get_vacc_per_available()

            # apply vacc_delay
            vacc_per_available = vacc_per_available.groupby(['region','age']).shift(vacc_delay).fillna(0)

            # group vacc_per_available by trange interval
            t_index_rounded_down_to_tslices = pd.cut(vacc_per_available.index.get_level_values('t'), self.params_trange + [self.tmax], right=False, retbins=False, labels=self.params_trange)
            vacc_per_available = vacc_per_available.groupby([t_index_rounded_down_to_tslices, 'region','age']).mean()

            # convert to dictionaries for performance lookup
            vacc_per_available_dict = vacc_per_available.to_dict()

            # set the fail rate and vacc per unvacc rate for each dose
            for shot in self.attrs['vacc'][1:]:
                for age in self.attrs['age']:
                    for region in self.attrs['region']:
                        for t in self.params_trange:
                            self.set_param(f'{shot}_per_available', vacc_per_available_dict[shot][(t, region, age)], {'age': age, 'region':region}, trange=[t])

        # alter parameters based on timeseries effects
        multipliers = self.get_timeseries_effect_multipliers()
        for param, mult_by_t in multipliers.to_dict().items():
            for t, mult in mult_by_t.items():
                if t in self.params_trange:
                    self.set_param(param, mult=mult, trange=[t])

        # alter parameters based on attribute multipliers
        if self.attribute_multipliers:
            for attr_mult_specs in self.attribute_multipliers:
                self.set_param(**attr_mult_specs)


    # set the "non-linear multiplier" which is a scalar (for each t value) that will scale all non-linear flows
    # used for changing TC without rebuilding all the matrices
    def set_nonlinear_multiplier(self, mult, trange=None):
        trange = trange if trange is not None else self.params_trange
        for t in trange:
            self.nonlinear_multiplier[t] = mult

    # set TC by slice, and update non-linear multipliers; defaults to reseting the last TC values
    def apply_tc(self, tc=None, tslices=None, suppress_ode_rebuild=False):
        # if tslices are provided, replace any tslices >= tslices[0] with the new tslices
        if tslices is not None:
            self.tslices = [tslice for tslice in self.tslices if tslice < tslices[0]] + tslices
            self.params_trange = sorted(list(set(self.params_trange).union(self.tslices)))
            for i, t in enumerate(self.params_trange):
                if t not in self.params_by_t.keys():
                    self.params_by_t[t] = copy.deepcopy(self.params_by_t[self.params_trange[i - 1]])

            self.build_t_lookups()  # rebuild t lookups
            self.tc = self.tc[:len(self.tslices) + 1]  # truncate tc if longer than tslices
            self.tc += [self.tc[-1]] * (1 + len(self.tslices) - len(self.tc))  # extend tc if shorter than tslices

        # if tc is provided, replace the
        if tc is not None:
            self.tc = self.tc[:-len(tc)] + tc

        # if the lengths do not match, raise an error
        if len(self.tc) != len(self.tslices) + 1:
            raise ValueError(
                f'The length of tc ({len(self.tc)}) must be equal to the length of tslices ({len(self.tslices)}) + 1.')

        # apply the new TC values to the non-linear multiplier to update the ODE
        # TODO: only update the nonlinear multipliers for TCs that have been changed
        if not suppress_ode_rebuild:
            for tmin, tmax, tc in zip([self.tmin] + self.tslices, self.tslices + [self.tmax], self.tc):
                self.set_nonlinear_multiplier(1 - tc, trange=range(tmin, tmax))

    def build_t_lookups(self):
        # rebuild t lookups
        self.t_prev_lookup = {t_int: max(t for t in self.params_trange if t <= t_int) for t_int in range(min(self.params_trange), max(self.params_trange))}
        self.t_prev_lookup[max(self.params_trange)] = self.t_prev_lookup[max(self.params_trange) - 1]
        self.t_next_lookup = {t_int: min(t for t in self.params_trange if t > t_int) for t_int in range(min(self.params_trange), max(self.params_trange))}
        self.t_next_lookup[max(self.params_trange)] = self.t_next_lookup[max(self.params_trange) - 1]


    def build_ode_or_something(self):
        self.build_t_lookups()

        # set nonlinear multiplier
        for tmin, tmax, tc in zip([self.tmin] + self.tslices, self.tslices + [self.tmax], self.tc):
            self.set_nonlinear_multiplier(1 - tc, trange=range(tmin, tmax))


        self.compartments_as_index = pd.MultiIndex.from_product(self.attrs.values(), names=self.attrs.keys())
        self.compartments = list(self.compartments_as_index)
        self.cmpt_idx_lookup = pd.Series(index=self.compartments_as_index,
                                         data=range(len(self.compartments_as_index))).to_dict()
        self.length = len(self.cmpt_idx_lookup)
        self.param_compartments = list(set(tuple(
            attr_val for attr_val, attr_name in zip(cmpt, self.attr_names) if attr_name in self.param_attr_names) for cmpt in self.compartments))

    # assign default values to matrices
    def reset_ode(self):
        self.terms = []
        self.linear_matrix = {t: spsp.lil_matrix((self.length, self.length)) for t in self.params_trange}
        self.nonlinear_matrices = {t: defaultdict(lambda: spsp.lil_matrix((self.length, self.length))) for t in self.params_trange}
        self.constant_vector = {t: np.zeros(self.length) for t in self.params_trange}
        self.nonlinear_multiplier = {}

     # takes a symbolic equation, and looks up variable names in params to provide a computed output for each t in trange
    def calc_coef_by_t(self, coef, cmpt):
        if len(cmpt) > len(self.param_attr_names):
            param_cmpt = tuple(attr for attr, level in zip(cmpt, self.attr_names) if level in self.param_attr_names)
        else:
            param_cmpt = cmpt

        if isinstance(coef, dict):
            return {t: coef[t] if t in coef.keys() else 0 for t in self.params_trange}
        elif callable(coef):
            return {t: coef(t) for t in self.params_trange}
        elif isinstance(coef, str):
            if coef == '1':
                coef_by_t = {t: 1 for t in self.params_trange}
            else:
                coef_by_t = {}
                expr = parse_expr(coef)
                relevant_params = [str(s) for s in expr.free_symbols]
                param_cmpts_by_param = {**{param: param_cmpt for param in relevant_params}}
                if len(relevant_params) == 1 and coef == relevant_params[0]:
                    coef_by_t = {t: self.params_by_t[t][param_cmpt][coef] for t in self.params_trange}
                else:
                    func = sym.lambdify(relevant_params, expr)
                    for t in self.params_trange:
                        coef_by_t[t] = func(**{param: self.params_by_t[t][param_cmpts_by_param[param]][param] for param in relevant_params})
            return coef_by_t
        else:
            return {t: coef for t in self.params_trange}

    # add a flow term, and add new flow to ODE matrices
    def add_flow(self, from_cmpt, to_cmpt, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None):
        if len(from_cmpt) < len(self.attrs.keys()):
            raise ValueError(f'Origin compartment `{from_cmpt}` does not have the right number of attributes.')
        if len(to_cmpt) < len(self.attrs.keys()):
            raise ValueError(f'Destination compartment `{to_cmpt}` does not have the right number of attributes.')
        if scale_by_cmpts is not None:
            for cmpt in scale_by_cmpts:
                if len(cmpt) < len(self.attrs.keys()):
                    raise ValueError(f'Scaling compartment `{cmpt}` does not have the right number of attributes.')

        if coef is not None:
            if scale_by_cmpts_coef:
                coef_by_t_lookup = {c: self.calc_coef_by_t(c, to_cmpt) for c in set(scale_by_cmpts_coef)}
                coef_by_t_ld = [coef_by_t_lookup[c] for c in scale_by_cmpts_coef]
                coef_by_t_dl = {t: [dic[t] for dic in coef_by_t_ld] for t in self.params_trange}
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

        if not (isinstance(term, ConstantODEFlowTerm)) and all([c == 0 for c in term.coef_by_t.values()]):
            pass
        elif isinstance(term, ConstantODEFlowTerm) and all([c == 0 for c in term.constant_by_t.values()]):
            pass
        else:
            self.terms.append(term)

            # add term to matrices
            for t in self.params_trange:
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

    # build ODE
    def build_ode_flows(self):
        self.reset_ode()

        # vaccination
        # first shot
        self.add_flows_by_attr({'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'weak'}, coef=f'shot1_per_available * (1 - shot1_fail_rate)')
        self.add_flows_by_attr({'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'none'}, coef=f'shot1_per_available * shot1_fail_rate')
        # second and third shot
        for i in [2, 3]:
            for immun in self.attrs['immun']:
                # if immun is none, that means that the first vacc shot failed, which means that future shots may fail as well
                if immun == 'none':
                    self.add_flows_by_attr({'vacc': f'shot{i-1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, coef=f'shot{i}_per_available * (1 - shot{i}_fail_rate / shot{i-1}_fail_rate)')
                    self.add_flows_by_attr({'vacc': f'shot{i-1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'none'}, coef=f'shot{i}_per_available * (shot{i}_fail_rate / shot{i-1}_fail_rate)')
                else:
                    self.add_flows_by_attr({'vacc': f'shot{i-1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, coef=f'shot{i}_per_available')

        # seed variants
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'none'}, constant='initial_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'alpha'}, constant='alpha_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'delta'}, constant='delta_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'omicron'}, constant='om_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'ba2'}, constant='ba2_seed')

        # exposure
        asymptomatic_transmission = '(1 - immunity) * kappa * betta / total_pop'
        for variant in self.attrs['variant']:
            # No mobility between regions (or a single region)
            if self.model_mobility_mode is None or self.model_mobility_mode == "none":
                for region in self.attrs['region']:
                    sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant, 'region': region})
                    asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant, 'region': region})
                    self.add_flows_by_attr({'seir': 'S', 'variant': 'none', 'region': region}, {'seir': 'E', 'variant': variant}, coef=f'lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
                    self.add_flows_by_attr({'seir': 'S', 'variant': 'none', 'region': region}, {'seir': 'E', 'variant': variant}, coef=asymptomatic_transmission, scale_by_cmpts=asympt_cmpts)
            # Transmission parameters attached to the susceptible population
            elif self.model_mobility_mode == "population_attached":
                for from_region in self.attrs['region']:
                    # kappa in this mobility mode is associated with the susceptible population, so no need to store every kappa in every region
                    asymptomatic_transmission = f'(1 - immunity) * kappa_pa * betta / region_pop_{from_region}'
                    for to_region in self.attrs['region']:
                        sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant, 'region': from_region})
                        asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant, 'region': from_region})
                        self.add_flows_by_attr({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_{from_region} * lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
                        self.add_flows_by_attr({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_{from_region} * {asymptomatic_transmission}', scale_by_cmpts=asympt_cmpts)
            # Transmission parameters attached to the transmission location
            elif self.model_mobility_mode == "location_attached":
                for from_region in self.attrs['region']:
                    for in_region in self.attrs['region']:
                        # kappa in this mobility mode is associated with the in_region, so need to store every kappa in every region
                        asymptomatic_transmission = f'(1 - immunity) * kappa_la_{in_region} * betta / region_pop_{from_region}'
                        for to_region in self.attrs['region']:
                            sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant, 'region': from_region})
                            asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant, 'region': from_region})
                            self.add_flows_by_attr({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_fracin_{in_region} * mob_{in_region}_fracfrom_{from_region} * lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
                            self.add_flows_by_attr({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_fracin_{in_region} * mob_{in_region}_fracfrom_{from_region} * {asymptomatic_transmission}', scale_by_cmpts=asympt_cmpts)

        # disease progression
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'I'}, coef='1 / alpha * pS')
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'A'}, coef='1 / alpha * (1 - pS)')
        self.add_flows_by_attr({'seir': 'I'}, {'seir': 'Ih'}, coef='gamm * hosp * (1 - severe_immunity)')

        # disease termination
        for variant in self.attrs['variant']:
            # TODO: Rename "non-omicron" to "other"; will need to make the change in attribute_multipliers, which will break old specifications
            priorinf = variant if variant != 'none' and variant in self.attrs['priorinf'] else 'non-omicron'
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='gamm * (1 - hosp - dnh) * (1 - priorinf_fail_rate)')
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * (1 - hosp - dnh) * priorinf_fail_rate')
            self.add_flows_by_attr({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='gamm * (1 - priorinf_fail_rate)')
            self.add_flows_by_attr({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * priorinf_fail_rate')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='1 / hlos * (1 - dh) * (1 - priorinf_fail_rate)')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * (1 - dh) * priorinf_fail_rate')
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * dnh * (1 - severe_immunity)')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * dh')

        # immunity decay
        self.add_flows_by_attr({'immun': 'strong'}, {'immun': 'weak'}, coef='1 / imm_decay_days')

    # convert ODE matrices to CSR format, to (massively) improve performance
    def compile(self):
        for t in self.params_trange:
            self.linear_matrix[t] = self.linear_matrix[t].tocsr()
            for k, v in self.nonlinear_matrices[t].items():
                self.nonlinear_matrices[t][k] = v.tocsr()

    # ODE step forward
    def ode(self, t, y):
        dy = [0] * self.length
        t_int = self.t_prev_lookup[math.floor(t)]

        # apply linear terms
        dy += (self.linear_matrix[t_int]).dot(y)

        # apply non-linear terms
        for scale_by_cmpt_idxs, matrix in self.nonlinear_matrices[t_int].items():
            dy += self.nonlinear_multiplier[t_int] * sum(itemgetter(*scale_by_cmpt_idxs)(y)) * (matrix).dot(
                y)

        # apply constant terms
        dy += self.constant_vector[t_int]

        return dy

    # solve ODE using scipy.solve_ivp, and put solution in solution_y and solution_ydf
    # TODO: try Julia ODE package, to improve performance
    def solve_ode(self, y0_dict, method='RK45', max_step=1.0):
        self.solution = spi.solve_ivp(
            fun=self.ode,
            t_span=[min(self.params_trange), max(self.params_trange)],
            y0=self.y0_from_dict(y0_dict),
            t_eval=self.trange,
            method=method,
            max_step=max_step)
        if not self.solution.success:
            raise RuntimeError(f'ODE solver failed with message: {self.solution.message}')
        self.solution_y = np.transpose(self.solution.y)
        self.solution_ydf = pd.concat([self.y_to_series(self.solution_y[t]) for t in self.trange], axis=1, keys=self.trange, names=['t']).transpose()

    # a model must be prepped before it can be run; if any params EXCEPT the efs (i.e. TC) change, it must be re-prepped
    def prep(self, rebuild_param_lookups=True, **build_param_lookup_args):
        if rebuild_param_lookups:
            self.build_param_lookups(**build_param_lookup_args)
        self.build_ode_flows()
        self.compile()

    # override solve_ode to use default y0_dict
    def solve_seir(self, method='RK45', y0_dict=None):
        y0_dict = y0_dict if y0_dict is not None else self.y0_dict
        self.solve_ode(y0_dict=y0_dict, method=method)
