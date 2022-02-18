import copy

import pandas as pd
import numpy as np
import datetime as dt
import json

import scipy.stats as sps
from sqlalchemy import MetaData

from covid_model.db import db_engine
from covid_model.data_imports import ExternalVaccWithProjections, ExternalVacc
from covid_model.utils import get_params


class CovidModelSpecifications:

    def __init__(self, start_date=dt.date(2020, 1, 24), end_date=dt.date(2022, 5, 31)):

        self.start_date = start_date
        self.end_date = end_date

        self.spec_id = None
        self.base_spec_id = None
        self.tags = {}

        self.tslices = None
        self.tc = None
        self.tc_cov = None

        self.model_params = None
        self.group_pops = None
        self.vacc_proj_params = None
        self.timeseries_effects = {}
        self.attribute_multipliers = None

        self.actual_vacc_df = None
        self.proj_vacc_df = None  # the combined vacc rate df (including proj) is saved to avoid unnecessary processing

    def copy(self, new_end_date=None):
        specs = CovidModelSpecifications(self.start_date, new_end_date if new_end_date is not None else self.end_date)

        specs.tags = self.tags.copy()
        specs.tc_cov = self.tc_cov.copy()
        specs.actual_vacc_df = self.actual_vacc_df.copy()
        specs.timeseries_effects = self.timeseries_effects.copy()
        specs.base_spec_id = self.spec_id

        specs.build(specs=specs, tslices=self.tslices, tc=self.tc,
                      params=self.model_params,
                      vacc_proj_params=self.vacc_proj_params,
                      attribute_multipliers=self.attribute_multipliers)

        return specs

    @classmethod
    def from_db(cls, engine, spec_id, new_end_date=None):
        df = pd.read_sql_query(f"select * from covid_model.specifications where spec_id = {spec_id}", con=engine, coerce_float=True)
        if len(df) == 0:
            raise ValueError(f'{spec_id} is not a valid spec ID.')
        row = df.iloc[0]

        specs = CovidModelSpecifications(start_date=row['start_date'], end_date=new_end_date if new_end_date is not None else row['end_date'])
        specs.spec_id = row['spec_id']
        specs.base_spec_id = row['base_spec_id']

        specs.set_tc(tslices=row['tslices'], tc=row['tc'], tc_cov=row['tc_cov'])
        specs.set_model_params(row['model_params'])

        specs.actual_vacc_df = pd.concat({k: pd.DataFrame(v).stack() for k, v in row['vacc_actual'].items()}, axis=1).rename_axis(index=['t', 'age'])
        specs.set_vacc_proj(row['vacc_proj_params'])

        specs.timeseries_effects = row['timeseries_effects']
        specs.attribute_multipliers = row['attribute_multipliers']

        return specs

    @classmethod
    def build(cls, specs=None, engine=None,
              start_date=None, end_date=None,
              tslices=None, tc=None, params=None,
              refresh_actual_vacc=False, vacc_proj_params=None,
              timeseries_effect_multipliers=None, variant_prevalence=None, mab_prevalence=None,
              attribute_multipliers=None):

        if specs is None:
            specs = CovidModelSpecifications(start_date=start_date, end_date=end_date)
        elif isinstance(specs, (int, np.int64)):
            specs = CovidModelSpecifications.from_db(engine, specs, new_end_date=end_date)

        if tslices or tc:
            specs.set_tc(tslices, tc)
        if params:
            specs.set_model_params(params)
        if refresh_actual_vacc:
            specs.set_actual_vacc(engine)
        if refresh_actual_vacc or vacc_proj_params:
            specs.set_vacc_proj(vacc_proj_params)

        # TODO: make add_timeseries_effect use existing prevalence and existing multipliers if new prevalence or mults are not provided
        # if variant_prevalence or param_multipliers:
        if variant_prevalence:
            specs.add_timeseries_effect('variant', prevalence_data=variant_prevalence,
                                                      param_multipliers=timeseries_effect_multipliers,
                                                      fill_forward=True)
        # if mab_prevalence or param_multipliers:
        if mab_prevalence:
            specs.add_timeseries_effect('mab', prevalence_data=mab_prevalence,
                                                      param_multipliers=timeseries_effect_multipliers,
                                                      fill_forward=True)

        if attribute_multipliers:
            specs.set_attr_mults(attribute_multipliers)

        return specs

    def write_to_db(self, engine, schema='covid_model', table='specifications', tags=None):
        metadata = MetaData(schema=schema)
        metadata.reflect(engine, only=['specifications'])
        specs_table = metadata.tables[f'{schema}.{table}']

        if tags is not None:
            self.tags.update(tags)

        stmt = specs_table.insert().values(
            created_at=dt.datetime.now(),
            base_spec_id=int(self.base_spec_id),
            tags=self.tags,
            start_date=self.start_date,
            end_date=self.end_date,
            tslices=self.tslices,
            tc=self.tc,
            tc_cov=self.tc_cov,
            model_params=self.model_params,
            vacc_actual={dose: rates.unstack(level='age').to_dict(orient='list') for dose, rates in self.actual_vacc_df.to_dict(orient='series').items()},
            vacc_proj_params=self.vacc_proj_params,
            vacc_proj={dose: rates.unstack(level='age').to_dict(orient='list') for dose, rates in self.proj_vacc_df.to_dict(orient='series').items()} if self.proj_vacc_df is not None else None,
            timeseries_effects=self.timeseries_effects,
            attribute_multipliers=self.attribute_multipliers
        )

        conn = engine.connect()
        result = conn.execute(stmt)

        self.spec_id = result.inserted_primary_key[0]

    @property
    def days(self):
        return (self.end_date - self.start_date).days

    def set_tc(self, tslices, tc, tc_cov=None, append=False):
        self.tslices = (self.tslices if append else []) + list(tslices)
        self.tc = (self.tc if append else []) + list(tc)
        self.tc_cov = [list(a) for a in tc_cov] if tc_cov is not None else None

    def set_model_params(self, model_params):
        self.model_params = copy.deepcopy(model_params) if type(model_params) == dict else json.load(open(model_params))
        self.group_pops = self.model_params['group_pop']

    def set_vacc_proj(self, vacc_proj_params=None):
        if vacc_proj_params is not None:
            self.vacc_proj_params = copy.deepcopy(vacc_proj_params) if isinstance(vacc_proj_params, dict) else json.load(open(vacc_proj_params))
        self.proj_vacc_df = self.get_proj_vacc()

    def set_actual_vacc(self, engine=None, actual_vacc_df=None):
        if engine is not None:
            self.actual_vacc_df = ExternalVacc(engine, t0_date=self.start_date).fetch()
        if actual_vacc_df is not None:
            self.actual_vacc_df = actual_vacc_df.copy()

    def set_attr_mults(self, attr_mults):
        self.attribute_multipliers = copy.deepcopy(attr_mults) if isinstance(attr_mults, list) else json.load(open(attr_mults))

    def add_timeseries_effect(self, effect_type_name, prevalence_data, param_multipliers, fill_forward=False):
        # build prevalence and multiplier dataframes from inputs
        prevalence_df = pd.read_csv(prevalence_data, parse_dates=['date'], index_col=0) if isinstance(prevalence_data, str) else prevalence_data.copy()
        prevalence_df = prevalence_df[prevalence_df.max(axis=1) > 0]
        if fill_forward and self.end_date > prevalence_df.index.max().date():
            projections = pd.DataFrame.from_dict({date: prevalence_df.iloc[-1] for date in pd.date_range(prevalence_df.index.max() + dt.timedelta(days=1), self.end_date)}, orient='index')
            prevalence_df = pd.concat([prevalence_df, projections]).sort_index()

        multiplier_dict = json.load(open(param_multipliers)) if isinstance(param_multipliers, str) else copy.deepcopy(param_multipliers)

        self.timeseries_effects[effect_type_name] = []
        for effect_name in prevalence_df.columns:
            d = {'effect_name': effect_name, 'multipliers': multiplier_dict[effect_name], 'start_date': prevalence_df.index.min().strftime('%Y-%m-%d'), 'prevalence': list(prevalence_df[effect_name].values)}
            self.timeseries_effects[effect_type_name].append(d)

    def get_timeseries_effect_multipliers(self):
        params = set().union(*[effect_specs['multipliers'].keys() for effects in self.timeseries_effects.values() for effect_specs in effects])
        multipliers = pd.DataFrame(
            index=pd.date_range(self.start_date, self.end_date),
            columns=params,
            data=1.0
        )

        multiplier_dict = {}
        for effect_type in self.timeseries_effects.keys():
            prevalence_df = pd.DataFrame(index=pd.date_range(self.start_date, self.end_date))

            for effect_specs in self.timeseries_effects[effect_type]:
                if len(effect_specs['start_date']) == 8:
                    effect_specs['start_date'] = '20' + effect_specs['start_date']
                start_date = dt.datetime.strptime(effect_specs['start_date'], '%Y-%m-%d').date()
                end_date = self.end_date

                if start_date < end_date:
                    prevalence = effect_specs['prevalence']
                    prevalence = prevalence[:(end_date - start_date).days]
                    while len(prevalence) < (end_date - start_date).days:
                        prevalence.append(prevalence[-1])

                    prevalence_df[effect_specs['effect_name']] = 0
                    prevalence_df.loc[start_date:(end_date - dt.timedelta(days=1)), effect_specs['effect_name']] = prevalence
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

    def get_proj_vacc(self):
        proj_lookback = self.vacc_proj_params['lookback'] if 'lookback' in self.vacc_proj_params.keys() else 7
        proj_fixed_rates = self.vacc_proj_params['fixed_rates'] if 'fixed_rates' in self.vacc_proj_params.keys() else None
        max_cumu = self.vacc_proj_params['max_cumu'] if 'max_cumu' in self.vacc_proj_params.keys() else 0
        max_rate_per_remaining = self.vacc_proj_params['max_rate_per_remaining'] if 'max_rate_per_remaining' in self.vacc_proj_params.keys() else 1.0
        realloc_priority = self.vacc_proj_params['realloc_priority'] if 'realloc_priority' in self.vacc_proj_params.keys() else None

        shots = list(self.actual_vacc_df.columns)

        # add projections
        proj_from_t = self.actual_vacc_df.index.get_level_values('t').max() + 1
        proj_to_t = (self.end_date - self.start_date).days
        if proj_to_t > proj_from_t:
            proj_trange = range(proj_from_t, proj_to_t)
            # project rates based on the last {proj_lookback} days of data
            projected_rates = self.actual_vacc_df.loc[(proj_from_t - proj_lookback):].groupby('age').sum() / float(proj_lookback)
            # override rates using fixed values from proj_fixed_rates, when present
            if proj_fixed_rates:
                for shot in shots:
                    projected_rates[shot] = pd.DataFrame(proj_fixed_rates)[shot]
            # build projections
            projections = pd.concat({t: projected_rates for t in proj_trange}).rename_axis(index=['t', 'age'])

            # reduce rates to prevent cumulative vaccination from exceeding max_cumu
            if max_cumu:
                cumu_vacc = self.actual_vacc_df.groupby('age').sum()
                groups = realloc_priority if realloc_priority else projections.index.unique('age')
                # vaccs = df.index.unique('vacc')
                for t in projections.index.unique('t'):
                    this_max_cumu = get_params(max_cumu.copy(), t)
                    max_cumu_df = pd.DataFrame(this_max_cumu) * pd.DataFrame(self.group_pops, index=shots).transpose()
                    for i in range(len(groups)):
                        group = groups[i]
                        current_rate = projections.loc[(t, group)]
                        max_rate = max_rate_per_remaining * (max_cumu_df.loc[group] - cumu_vacc.loc[group])
                        excess_rate = (projections.loc[(t, group)] - max_rate).clip(lower=0)
                        projections.loc[(t, group)] -= excess_rate
                        # if a reallocate_order is provided, reallocate excess rate to other groups
                        if i < len(groups) - 1 and realloc_priority is not None:
                            projections.loc[(t, groups[i + 1])] += excess_rate

                    cumu_vacc += projections.loc[t]

            return projections

    def get_vacc_rates(self):
        df = pd.concat([self.actual_vacc_df, self.proj_vacc_df])
        return df

    def get_vacc_per_available(self):
        vacc_rates = self.get_vacc_rates()
        populations = pd.Series(self.model_params['group_pop'], name='population').rename_axis(index='age')
        cumu_vacc = vacc_rates.groupby('age').cumsum()
        cumu_vacc_final_shot = cumu_vacc - cumu_vacc.shift(-1, axis=1).fillna(0)
        cumu_vacc_final_shot['none'] = cumu_vacc_final_shot.join(populations)['population'] - cumu_vacc_final_shot.sum(axis=1)
        cumu_vacc_final_shot = cumu_vacc_final_shot.reindex(columns=['none', 'shot1', 'shot2', 'shot3'])
        available_for_vacc = cumu_vacc_final_shot.shift(1, axis=1).drop(columns='none')

        return (vacc_rates / available_for_vacc).fillna(0)
