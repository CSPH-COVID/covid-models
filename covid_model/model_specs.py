import pandas as pd
import numpy as np
import datetime as dt
import json
import copy

import scipy.stats as sps
from sqlalchemy import MetaData, func
from sqlalchemy.orm import Session

from covid_model.db import db_engine, get_sqa_table
from covid_model.data_imports import ExternalVaccWithProjections, ExternalVacc
from covid_model.utils import get_params


class CovidModelSpecifications:

    def __init__(self, start_date=None, end_date=None, engine=None, from_specs=None, **spec_args):

        self.start_date = start_date
        self.end_date = None

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

        # base specs can be provided via a database spec_id or an CovidModelSpecifications object
        if from_specs is not None:
            # if from_specs is an int, get specs from the database
            if isinstance(from_specs, int):
                if engine is None:
                    raise ValueError(f'Database engine is required to fetch specification {from_specs} from db.')

                df = pd.read_sql_query(f"select * from covid_model.specifications where spec_id = {from_specs}", con=engine, coerce_float=True)
                if len(df) == 0:
                    raise ValueError(f'{from_specs} is not a valid spec ID.')
                row = df.iloc[0]

                self.start_date = row['start_date']
                self.end_date = end_date if end_date is not None else row['end_date']
                self.spec_id = row['spec_id']
                self.base_spec_id = row['base_spec_id']
                self.set_tc(tslices=row['tslices'], tc=row['tc'], tc_cov=json.loads(row['tc_cov'].replace('{', '[').replace('}', ']')))
                self.set_model_params(json.loads(row['model_params']))
                self.actual_vacc_df = pd.concat({k: pd.DataFrame(v).stack() for k, v in json.loads(row['vacc_actual']).items()}, axis=1).rename_axis(index=['t', 'age'])
                self.set_vacc_proj(json.loads(row['vacc_proj_params']))
                self.timeseries_effects = json.loads(row['timeseries_effects'])
                self.attribute_multipliers = json.loads(row['attribute_multipliers'])
            # if from_specs is an existing specification, do a deep copy
            elif isinstance(from_specs, CovidModelSpecifications):
                self.start_date = from_specs.start_date
                self.tags = copy.deepcopy(from_specs.tags)
                self.tc_cov = copy.deepcopy(from_specs.tc_cov)
                self.actual_vacc_df = copy.deepcopy(from_specs.actual_vacc_df)
                self.timeseries_effects = copy.deepcopy(from_specs.timeseries_effects)
                self.base_spec_id = from_specs.spec_id if from_specs.spec_id is not None else from_specs.base_spec_id
                self.update_specs(
                    end_date=end_date if end_date is not None else from_specs.end_date,
                    tslices=copy.deepcopy(from_specs.tslices), tc=copy.deepcopy(from_specs.tc),
                    params=copy.deepcopy(from_specs.model_params),
                    vacc_proj_params=copy.deepcopy(from_specs.vacc_proj_params),
                    attribute_multipliers=copy.deepcopy(from_specs.attribute_multipliers))
            else:
                raise TypeError(f'from_specs must be an int or CovidModelSpecifications; not a {type(from_specs)}.')

        if start_date is not None:
            if self.start_date is not None and start_date != self.start_date:
                raise NotImplementedError(f'Changing the start_date of an existing spec is not supported.')
            self.start_date = start_date

        self.update_specs(**spec_args)

    # def copy(self, new_end_date=None):
    #     specs = CovidModelSpecifications(
    #         start_date=self.start_date,
    #         end_date=new_end_date if new_end_date is not None else self.end_date,
    #         tslices=self.tslices, tc=self.tc,
    #         params=self.model_params,
    #         vacc_proj_params=self.vacc_proj_params,
    #         attribute_multipliers=self.attribute_multipliers
    #     )
    #
    #     specs.tags = self.tags.copy()
    #     specs.tc_cov = self.tc_cov.copy()
    #     specs.actual_vacc_df = self.actual_vacc_df.copy()
    #     specs.timeseries_effects = self.timeseries_effects.copy()
    #     specs.base_spec_id = self.spec_id
    #
    #     return specs

    def update_specs(self, engine=None, end_date=None,
                     tslices=None, tc=None, params=None,
                     refresh_actual_vacc=False, vacc_proj_params=None,
                     timeseries_effect_multipliers=None, variant_prevalence=None, mab_prevalence=None,
                     attribute_multipliers=None,
                     region_params=None, region=None):

        if end_date is not None:
            self.end_date = end_date
        if tslices or tc:
            self.set_tc(tslices=tslices, tc=tc)
        if params:
            self.set_model_params(params, region_params, region)
        if refresh_actual_vacc:
            self.set_actual_vacc(engine, county_ids=self.tags["county_fips"] if "county_fips" in self.tags.keys() else None)
        if refresh_actual_vacc or vacc_proj_params or end_date:
            if vacc_proj_params is not None:
                self.vacc_proj_params = vacc_proj_params
            self.set_vacc_proj(self.vacc_proj_params)

        # TODO: make add_timeseries_effect use existing prevalence and existing multipliers if new prevalence or mults are not provided
        # if variant_prevalence or param_multipliers:
        if variant_prevalence:
            self.add_timeseries_effect('variant', prevalence_data=variant_prevalence,
                                                      param_multipliers=timeseries_effect_multipliers,
                                                      fill_forward=True)
        # if mab_prevalence or param_multipliers:
        if mab_prevalence:
            self.add_timeseries_effect('mab', prevalence_data=mab_prevalence,
                                                      param_multipliers=timeseries_effect_multipliers,
                                                      fill_forward=True)

        if attribute_multipliers:
            self.set_attr_mults(attribute_multipliers)

    def write_to_db(self, engine, schema='covid_model', table='specifications', tags=None):
        specs_table = get_sqa_table(engine, schema=schema, table=table)

        if tags is not None:
            self.tags.update(tags)

        with Session(engine) as session:
            max_spec_id = session.query(func.max(specs_table.c.spec_id)).scalar()
            self.spec_id = max_spec_id + 1

            stmt = specs_table.insert().values(
                spec_id=self.spec_id,
                created_at=dt.datetime.now(),
                base_spec_id=int(self.base_spec_id) if self.base_spec_id is not None else None,
                tags=json.dumps(self.tags),
                start_date=self.start_date,
                end_date=self.end_date,
                tslices=self.tslices,
                tc=self.tc,
                tc_cov=json.dumps(self.tc_cov),
                model_params=json.dumps(self.model_params),
                vacc_actual=json.dumps({dose: rates.unstack(level='age').to_dict(orient='list') for dose, rates in
                                        self.actual_vacc_df.to_dict(orient='series').items()}),
                vacc_proj_params=json.dumps(self.vacc_proj_params),
                vacc_proj=json.dumps({dose: rates.unstack(level='age').to_dict(orient='list') for dose, rates in
                                      self.proj_vacc_df.to_dict(
                                          orient='series').items()} if self.proj_vacc_df is not None else None),
                timeseries_effects=json.dumps(self.timeseries_effects),
                attribute_multipliers=json.dumps(self.attribute_multipliers)
            )
            session.execute(stmt)
            session.commit()

    @property
    def days(self):
        return (self.end_date - self.start_date).days

    def set_tc(self, tslices, tc, tc_cov=None, append=False):
        self.tslices = (self.tslices if append else []) + list(tslices)
        self.tc = (self.tc if append else []) + list(tc)
        self.tc_cov = [list(a) for a in tc_cov] if tc_cov is not None else None

    def set_model_params(self, model_params, region_model_params=None, region=None):
        # model_params may be dictionary or path to json file which will be converted to json
        # region_model_params is the same, but will only be used if region != None. Contains region specific modifications to parameters
        # every key present in region_model_params will completely overwrite that entry in model_params
        model_params = copy.deepcopy(model_params) if type(model_params) == dict else json.load(open(model_params))
        if region is not None:
            region_model_params = region_model_params if type(region_model_params) == dict else json.load(open(region_model_params))
            model_params.update(region_model_params[region])
            self.tags['region'] = region   # record which option we ran with
            self.tags['county_fips'] = model_params['county_fips']
            self.tags['county_names'] = model_params['county_names']
            _ = model_params.pop('county_fips')   # remove from the parameters
            _ = model_params.pop('county_names')   # remove from the parameters
        self.model_params = model_params
        self.group_pops = self.model_params['group_pop']

    def set_vacc_proj(self, vacc_proj_params=None):
        if vacc_proj_params is not None:
            self.vacc_proj_params = copy.deepcopy(vacc_proj_params) if isinstance(vacc_proj_params, dict) else json.load(open(vacc_proj_params))
        self.proj_vacc_df = self.get_proj_vacc()

    def set_actual_vacc(self, engine, county_ids=None, actual_vacc_df=None):
        if engine is not None:
            self.actual_vacc_df = ExternalVacc(engine, t0_date=self.start_date).fetch(county_ids=county_ids)
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
