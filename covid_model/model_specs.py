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
        self.vacc_immun_params = None
        self.vacc_proj_params = None
        self.timeseries_effects = {}
        self.attr_mults = None

        self.actual_vacc_df = None
        self.proj_vacc_df = None  # the combined vacc rate df, including proj, is saved to avoid unnecessary processing

    def set_all(self, spec_id, tslices, tc, tc_cov, model_params, actual_vacc_df, vacc_proj_params, vacc_immun_params, timeseries_effects, base_spec_id=None, tags={}):
        self.spec_id = spec_id
        self.base_spec_id = base_spec_id
        self.tags = tags

        self.set_tc(tslices, tc)
        self.set_model_params(model_params)
        self.tc_cov = tc_cov

        self.actual_vacc_df = actual_vacc_df
        self.set_vacc_proj(vacc_proj_params)
        self.set_vacc_immun(vacc_immun_params)

        self.timeseries_effects = timeseries_effects

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

        # specs.actual_vacc_df = pd.concat({k: pd.DataFrame(v) for k, v in row['vacc_actual'].items()}, axis=0).unstack(0).stack(0).rename_axis(index=['t', 'age'])
        specs.actual_vacc_df = pd.concat({k: pd.DataFrame(v).stack() for k, v in row['vacc_actual'].items()}, axis=1).rename_axis(index=['t', 'age'])
        specs.set_vacc_proj(row['vacc_proj_params'])
        specs.set_vacc_immun(row['vacc_immun_params'])

        specs.timeseries_effects = row['timeseries_effects']

        return specs

    def copy(self, new_end_date=None):
        specs = CovidModelSpecifications(self.start_date, new_end_date if new_end_date is not None else self.end_date)

        specs.set_all(spec_id=None, base_spec_id=self.spec_id, tags=self.tags, tslices=self.tslices, tc=self.tc, tc_cov=self.tc_cov,
                      model_params=self.model_params, actual_vacc_df=self.actual_vacc_df,
                      vacc_proj_params=self.vacc_proj_params,
                      vacc_immun_params=self.vacc_immun_params, timeseries_effects=self.timeseries_effects)

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
            vacc_immun_params=self.vacc_immun_params,
            timeseries_effects=self.timeseries_effects,
        )

        conn = engine.connect()
        result = conn.execute(stmt)

        self.spec_id = result.inserted_primary_key[0]

        # if self.model is not None:
        #     self.model.fit_id = result.inserted_primary_key[0]

        # return result.inserted_primary_key[0]

    @property
    def days(self):
        return (self.end_date - self.start_date).days

    def set_tc(self, tslices, tc, tc_cov=None, append=False):
        self.tslices = (self.tslices if append else []) + list(tslices)
        self.tc = (self.tc if append else []) + list(tc)
        self.tc_cov = [list(a) for a in tc_cov] if tc_cov is not None else None

    def set_model_params(self, model_params):
        self.model_params = model_params if type(model_params) == dict else json.load(open(model_params))
        self.group_pops = self.model_params['group_pop']

    def set_vacc_proj(self, vacc_proj_params=None):
        if vacc_proj_params is not None:
            self.vacc_proj_params = vacc_proj_params if isinstance(vacc_proj_params, dict) else json.load(open(vacc_proj_params))
        self.proj_vacc_df = self.get_proj_vacc()

    def set_actual_vacc(self, engine):
        # vacc_rate_df = ExternalVaccWithProjections(engine, t0_date=self.start_date, fill_to_date=self.end_date).fetch(proj_params=self.vacc_proj_params, group_pop=self.model_params['group_pop'])
        self.actual_vacc_df = ExternalVacc(engine, t0_date=self.start_date).fetch()
        # self.actual_vacc_df = ExternalVacc(engine).fetch()

    def set_vacc_immun(self, vacc_immun_params):
        self.vacc_immun_params = vacc_immun_params if isinstance(vacc_immun_params, dict) else json.load(open(vacc_immun_params))

    def set_attr_mults(self, attr_mults):
        self.attr_mults = attr_mults if isinstance(attr_mults, dict) else json.load(open(attr_mults))

    def add_timeseries_effect(self, effect_type_name, prevalence_data, param_multipliers, fill_forward=False):
        # build prevalence and multiplier dataframes from inputs
        prevalence_df = pd.read_csv(prevalence_data, parse_dates=['date'], index_col=0) if isinstance(prevalence_data, str) else prevalence_data.copy()
        prevalence_df = prevalence_df[prevalence_df.max(axis=1) > 0]
        if fill_forward and self.end_date > prevalence_df.index.max().date():
            projections = pd.DataFrame.from_dict({date: prevalence_df.iloc[-1] for date in pd.date_range(prevalence_df.index.max() + dt.timedelta(days=1), self.end_date)}, orient='index')
            prevalence_df = pd.concat([prevalence_df, projections]).sort_index()
        # prevalence_df.index = (prevalence_df.index.to_series() - self.min_date).dt.days

        multiplier_dict = json.load(open(param_multipliers)) if isinstance(param_multipliers, str) else param_multipliers
        # multiplier_df = pd.DataFrame.from_dict(multiplier_dict, orient='index').rename_axis(index='effect').fillna(1)

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
        by_last_shot = cumu_vacc - cumu_vacc.shift(-1, axis=1).fillna(0)
        by_last_shot['none'] = by_last_shot.join(populations)['population'] - by_last_shot.sum(axis=1)
        by_last_shot = by_last_shot.reindex(columns=['none', 'shot1', 'shot2', 'shot3'])
        available_for_vacc = by_last_shot.shift(1, axis=1).drop(columns='none')

        # cumu_vacc = cumu_vacc.join(populations)
        # cumu_vacc['none'] = cumu_vacc['population'] - cumu_vacc['shot1']
        # cumu_vacc = cumu_vacc.reindex(columns=['none', 'shot1', 'shot2', 'shot3'])
        # available_for_vacc = cumu_vacc.shift(1, axis=1).drop(columns='none')

        return (vacc_rates / available_for_vacc).fillna(0)

    def get_vacc_rate_per_unvacc(self):
        # calculate the vaccination rate per unvaccinated
        vacc_df = self.get_vacc_rates()
        cumu = vacc_df.groupby('age').cumsum()
        age_group_pop = vacc_df.index.get_level_values('age').to_series(index=vacc_df.index).replace(self.group_pops)
        unvacc = cumu.groupby('age').shift(1).fillna(0).apply(lambda s: age_group_pop - s)
        return vacc_df / unvacc

    def get_vacc_fail_per_vacc(self):
        return {k: v['fail_rate'] for k, v in self.vacc_immun_params.items()}

    def get_vacc_fail_reduction_per_vacc_fail(self, delay=7):
        vacc_fail_per_vacc_df = pd.DataFrame.from_dict(self.get_vacc_fail_per_vacc()).rename_axis(index='age')

        rate = self.get_vacc_rates().groupby(['age']).shift(delay).fillna(0)
        fail_increase = rate['shot1'] * vacc_fail_per_vacc_df['shot1']
        fail_reduction_per_vacc = vacc_fail_per_vacc_df.shift(1, axis=1) - vacc_fail_per_vacc_df
        fail_reduction = (rate * fail_reduction_per_vacc.fillna(0)).sum(axis=1)
        fail_cumu = (fail_increase - fail_reduction).groupby('age').cumsum()
        return (fail_reduction / fail_cumu).fillna(0)

    @classmethod
    def vacc_eff_decay_mult(cls, days_ago, delay, k):
        return (1.0718 * (1 - np.exp(-(days_ago + delay) / 7)) * np.exp(-days_ago / 540)) ** k

    def get_vacc_mean_efficacy(self, delay=7, k=1):
        if isinstance(k, str):
            k = self.model_params[k]

        vacc_df = self.get_vacc_rates()
        shots = list(vacc_df.columns)
        rate = vacc_df.groupby(['age']).shift(delay).fillna(0)
        cumu = rate.groupby(['age']).cumsum()
        vacc_effs = {k: v['eff'] for k, v in self.vacc_immun_params.items()}

        terminal_cumu_eff = pd.DataFrame(index=rate.index, columns=shots, data=0)
        for shot, next_shot in zip(shots, shots[1:] + [None]):
            nonzero_ts = rate[shot][rate[shot] > 0].index.get_level_values('t')
            if len(nonzero_ts) > 0:
                days_ago_range = range(nonzero_ts.max() - nonzero_ts.min() + 1)
                for days_ago in days_ago_range:
                    terminal_rate = np.minimum(rate[shot], (cumu[shot] - cumu.groupby('age').shift(-days_ago)[next_shot]).clip(lower=0)) if next_shot is not None else rate[shot]
                    terminal_cumu_eff[shot] += vacc_effs[shot] * self.vacc_eff_decay_mult(days_ago, delay, k) * terminal_rate.groupby(['age']).shift(days_ago).fillna(0)

        return (terminal_cumu_eff.sum(axis=1) / cumu[shots[0]]).fillna(0)
