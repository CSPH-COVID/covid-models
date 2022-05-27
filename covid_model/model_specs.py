### Python Standard Library ###
from operator import itemgetter
import datetime as dt
import json
import copy
from collections import OrderedDict
### Third Party Imports ###
import pandas as pd
import numpy as np
from scipy import sparse
from sqlalchemy import func
from sqlalchemy.orm import Session
### Local Imports ###
from covid_model.db import get_sqa_table
from covid_model.data_imports import ExternalVacc, get_region_mobility_from_db
from covid_model.utils import get_params


class CovidModelSpecifications:
    def __init__(self, start_date=None, engine=None, from_specs=None, **spec_args):
        self.max_step = None

        self.regions = None
        self.start_date = None
        self.end_date = None

        self.spec_id = None
        self.base_spec_id = None
        self.tags = {}
        self.result_id = None

        self.tslices = None
        self.tc = None
        self.tc_cov = None

        self.model_params = None
        self.vacc_proj_params = None
        self.timeseries_effects = {}
        self.attribute_multipliers = None

        self.model_region_definitions = None
        self.model_mobility_mode = None
        self.actual_mobility = {}
        self.mobility_proj_params = None

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

                self.regions = json.loads(row['regions'])
                self.start_date = row['start_date']
                self.end_date = spec_args['end_date'] if 'end_date' in spec_args and spec_args['end_date'] is not None else row['end_date']
                self.spec_id = row['spec_id']
                self.base_spec_id = row['base_spec_id']
                self.set_tc(tslices=row['tslices'], tc=row['tc'], tc_cov=json.loads(row['tc_cov'].replace('{', '[').replace('}', ']')))
                self.set_model_params(json.loads(row['model_params']))
                self.set_region_definitions(json.loads(row['region_definitions']))
                def vacc_load_fun(v):
                    df = pd.DataFrame(v).stack().reset_index(1)
                    spl = df['level_1'].str.split(";", expand=True)
                    df['region'] = spl[0]
                    df['age'] = spl[1]
                    return df.rename_axis(index='t').set_index(['region', 'age'], append=True).drop(columns='level_1').loc[:,0]
                self.actual_vacc_df = pd.concat({k: vacc_load_fun(v) for k, v in json.loads(row['vacc_actual']).items()}, axis=1)
                self.set_vacc_proj(json.loads(row['vacc_proj_params']))
                self.timeseries_effects = json.loads(row['timeseries_effects'])
                self.attribute_multipliers = json.loads(row['attribute_multipliers'])
                self.tags = json.loads(row['tags'])
                self.set_actual_mobility(actual_mobility=json.loads(row['mobility_actual']))
                self.set_mobility_proj(json.loads(row['mobility_proj_params']))
                self.model_mobility_mode = row['mobility_mode']
            # if from_specs is an existing specification, do a deep copy
            elif isinstance(from_specs, CovidModelSpecifications):
                self.start_date = from_specs.start_date
                self.tags = copy.deepcopy(from_specs.tags)
                self.tc_cov = copy.deepcopy(from_specs.tc_cov)
                self.actual_vacc_df = copy.deepcopy(from_specs.actual_vacc_df)
                self.timeseries_effects = copy.deepcopy(from_specs.timeseries_effects)
                self.base_spec_id = from_specs.spec_id if from_specs.spec_id is not None else from_specs.base_spec_id
                self.update_specs(
                    end_date=spec_args['end_date'] if 'end_date' in spec_args and spec_args['end_date'] is not None else from_specs.end_date,
                    tslices=copy.deepcopy(from_specs.tslices), tc=copy.deepcopy(from_specs.tc),
                    params=copy.deepcopy(from_specs.model_params),
                    vacc_proj_params=copy.deepcopy(from_specs.vacc_proj_params),
                    attribute_multipliers=copy.deepcopy(from_specs.attribute_multipliers),
                    region_definitions=from_specs.model_region_definitions,
                    regions=from_specs.regions,
                    mobility_proj_params=copy.deepcopy(from_specs.mobility_proj_params),
                    mobility_mode=from_specs.model_mobility_mode
                )
            else:
                raise TypeError(f'from_specs must be an int or CovidModelSpecifications; not a {type(from_specs)}.')

        if start_date is not None:
            if self.start_date is not None and start_date != self.start_date:
                raise NotImplementedError(f'Changing the start_date of an existing spec is not supported.')
            self.start_date = start_date

        self.update_specs(engine=engine, **spec_args)

    def update_specs(self, engine=None, end_date=None,
                     tslices=None, tc=None, params=None,
                     refresh_actual_vacc=False, vacc_proj_params=None,
                     timeseries_effect_multipliers=None, mab_prevalence=None,
                     attribute_multipliers=None,
                     region_definitions=None, regions=None,
                     mobility_mode='none', refresh_actual_mobility=False, mobility_proj_params=None,
                     region_fit_spec_ids=None, region_fit_result_ids=None, max_step=np.inf, **unused_args):
        self.max_step = max_step

        if regions:
            self.regions = regions
        self.model_mobility_mode = mobility_mode

        if end_date is not None:
            self.end_date = end_date
        if self.end_date is None:
            self.end_date = dt.date.today()
        if tslices or tc or end_date:
            self.set_tc(tslices=tslices, tc=tc)
        if params:
            self.set_model_params(params)
        if region_definitions:
            self.set_region_definitions(region_definitions)
        if refresh_actual_vacc:
            self.set_actual_vacc(engine)
        if refresh_actual_vacc or vacc_proj_params or end_date:
            if vacc_proj_params is not None:
                self.vacc_proj_params = vacc_proj_params
            self.set_vacc_proj(self.vacc_proj_params)

        # TODO: make add_timeseries_effect use existing multipliers if new mults are not provided (still a todo? -af)
        if mab_prevalence:
            self.add_timeseries_effect('mab', prevalence_data=mab_prevalence, param_multipliers=timeseries_effect_multipliers, fill_forward=True)

        if attribute_multipliers:
            self.set_attr_mults(attribute_multipliers)

        if refresh_actual_mobility:
            self.set_actual_mobility(engine)

        if mobility_proj_params or end_date:
            if mobility_proj_params is not None:
                self.mobility_proj_params = mobility_proj_params
            self.set_mobility_proj(self.mobility_proj_params)
        # add mobility data to model params.
        if self.model_mobility_mode != "none" and (refresh_actual_mobility or mobility_proj_params or end_date):
            self.model_params.update(self.get_mobility_as_params())

        if region_fit_spec_ids:
            self.model_params.update(self.get_kappas_as_params_using_region_fits(engine, region_fit_spec_ids, region_fit_result_ids))

    # handy properties for the beginning t, end t, and the full range of t values
    @property
    def tmin(self): return 0

    @property
    def tmax(self): return (self.end_date - self.start_date).days + 1

    @property
    def daterange(self): return pd.date_range(self.start_date, end=self.end_date - dt.timedelta(days=1))

    @property
    def tslices_dates(self): return [self.start_date + dt.timedelta(days=ts) for ts in [0] + self.tslices]

    def prepare_write_specs_query(self, tags=None):
        # returns all the data you would need to write to the database but doesn't actually write to the database
        if tags is not None:
            self.tags.update(tags)

        write_info = OrderedDict([
            ("created_at", dt.datetime.now()),
            ("base_spec_id", int(self.base_spec_id) if self.base_spec_id is not None else None),
            ("tags", json.dumps(self.tags)),
            ("start_date", self.start_date),
            ("end_date", self.end_date),
            ("tslices", self.tslices),
            ("tc", self.tc),
            ("tc_cov", json.dumps(self.tc_cov.tolist() if isinstance(self.tc_cov, np.ndarray) else self.tc_cov) if self.tc_cov is not None else None),
            ("model_params", json.dumps(self.model_params)),
            ("vacc_actual", json.dumps({dose: {";".join(key): val for key, val in rates.unstack(level=['region', 'age']).to_dict(orient='list').items()} for dose, rates in self.actual_vacc_df.to_dict(orient='series').items()})),
            ("vacc_proj_params", json.dumps(self.vacc_proj_params)),
            ("vacc_proj", json.dumps({dose: {";".join(key): val for key, val in rates.unstack(level=['region', 'age']).to_dict(orient='list').items()} for dose, rates in self.proj_vacc_df.to_dict(orient='series').items()} if self.proj_vacc_df is not None else None)),
            ("timeseries_effects", json.dumps(self.timeseries_effects)),
            ("attribute_multipliers", json.dumps(self.attribute_multipliers)),
            ("region_definitions", json.dumps(self.model_region_definitions)),
            ("mobility_actual", json.dumps(self.actual_mobility)),
            ("mobility_proj_params", json.dumps(self.mobility_proj_params)),
            ("mobility_proj", json.dumps(self.proj_mobility)),
            ("mobility_mode", self.model_mobility_mode),
            ("regions", json.dumps(self.regions))
        ])

        return write_info

    @classmethod
    def write_prepared_specs_to_db(cls, write_info, engine, spec_id: int=None):
        # writes the given info to the db without needing an explicit instance
        specs_table = get_sqa_table(engine, schema='covid_model', table='specifications')

        with Session(engine) as session:
            if spec_id is None:
                max_spec_id = session.query(func.max(specs_table.c.spec_id)).scalar()
                spec_id = max_spec_id + 1

            stmt = specs_table.insert().values(
                spec_id=spec_id,
                **write_info
            )
            session.execute(stmt)
            session.commit()
        return {**write_info, 'spec_id':spec_id}

    def write_specs_to_db(self, engine, tags=None):
        # get write info
        write_info = self.prepare_write_specs_query(tags=tags)

        specs_table = get_sqa_table(engine, schema='covid_model', table='specifications')
        with Session(engine) as session:
            # generate a spec_id so we can assign it to ourselves
            max_spec_id = session.query(func.max(specs_table.c.spec_id)).scalar()
            self.spec_id = max_spec_id + 1
        return self.write_prepared_specs_to_db(write_info, engine, spec_id=self.spec_id)

    @property
    def days(self):
        return (self.end_date - self.start_date).days

    def set_tc(self, tslices, tc, tc_cov=None, append=False):
        if tslices:
            self.tslices = (self.tslices if append else []) + list(tslices)
        if tc:
            self.tc = (self.tc if append else []) + list(tc)
        if tc_cov:
            self.tc_cov = [list(a) for a in tc_cov] if tc_cov is not None else None

        self.tslices = [ts for ts in self.tslices if ts < self.tmax]
        self.tc = self.tc[:len(self.tslices) + 1]

    def set_model_params(self, model_params):
        # model_params and model_region_definitions may be dictionary or path to json file which will be converted to json
        model_params = copy.deepcopy(model_params) if type(model_params) == dict else json.load(open(model_params))
        self.model_params = model_params

    def set_region_definitions(self, model_region_definitions):
        model_region_definitions = copy.deepcopy(model_region_definitions) if type(model_region_definitions) == dict else json.load(open(model_region_definitions))
        self.model_region_definitions = model_region_definitions

    def get_all_county_fips(self, regions=None):
        regions = self.regions if regions is None else regions
        regions = [regions] if isinstance(regions, str) else regions
        return [county_fips for region in regions for county_fips in self.model_region_definitions[region]['counties_fips'] for region in regions]

    def set_vacc_proj(self, vacc_proj_params=None):
        if vacc_proj_params is not None:
            self.vacc_proj_params = copy.deepcopy(vacc_proj_params) if isinstance(vacc_proj_params, dict) else json.load(open(vacc_proj_params))
        self.proj_vacc_df = self.get_proj_vacc()

    def set_actual_vacc(self, engine, actual_vacc_df=None):
        if engine is not None:
            actual_vacc_df_list = []
            for region in self.regions:
                county_ids = self.model_region_definitions[region]['counties_fips']
                actual_vacc_df_list.append(ExternalVacc(engine, t0_date=self.start_date).fetch(county_ids=county_ids).assign(region=region).set_index('region', append=True).reorder_levels(['t', 'region', 'age']))
            self.actual_vacc_df = pd.concat(actual_vacc_df_list)
        if actual_vacc_df is not None:
            self.actual_vacc_df = actual_vacc_df.copy()

    def set_actual_mobility(self, engine=None, actual_mobility=None):
        if engine is not None:
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
                dwell = sparse.coo_array((vals, (idx_i, idx_j)), shape=(len(regions), len(regions))).todense()
                dwell[np.isnan(dwell)] = 0
                dwell_rownorm = dwell / dwell.sum(axis=1)[:, np.newaxis]
                dwell_colnorm = dwell / dwell.sum(axis=0)[np.newaxis, :]
                dwell_matrices[t] = {"dwell": dwell.tolist(), "dwell_rownorm": dwell_rownorm.tolist(), "dwell_colnorm": dwell_colnorm.tolist()}

            self.actual_mobility = dwell_matrices
        if actual_mobility is not None:
            self.actual_mobility = actual_mobility.copy()

    def set_mobility_proj(self, mobility_proj_params=None):
        if mobility_proj_params is not None:
            self.mobility_proj_params = copy.deepcopy(mobility_proj_params) if isinstance(mobility_proj_params, dict) else json.load(open(mobility_proj_params))
        self.proj_mobility = self.get_proj_mobility()

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

    def get_proj_vacc(self):
        proj_lookback = self.vacc_proj_params['lookback'] if 'lookback' in self.vacc_proj_params.keys() else 7
        proj_fixed_rates = self.vacc_proj_params['fixed_rates'] if 'fixed_rates' in self.vacc_proj_params.keys() else None
        max_cumu = self.vacc_proj_params['max_cumu'] if 'max_cumu' in self.vacc_proj_params.keys() else 0
        max_rate_per_remaining = self.vacc_proj_params['max_rate_per_remaining'] if 'max_rate_per_remaining' in self.vacc_proj_params.keys() else 1.0
        realloc_priority = self.vacc_proj_params['realloc_priority'] if 'realloc_priority' in self.vacc_proj_params.keys() else None

        shots = list(self.actual_vacc_df.columns)
        region_df = pd.DataFrame({'region': self.regions})

        # add projections
        proj_from_t = self.actual_vacc_df.index.get_level_values('t').max() + 1
        proj_to_t = (self.end_date - self.start_date).days + 1
        if proj_to_t > proj_from_t:
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
                populations = [{'age': li['attributes']['age'], 'region': li['attributes']['region'], 'population': li['values']} for li in self.model_params['group_pop'] if li['attributes']['region'] in self.regions]
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

            return projections

    def get_vacc_rates(self):
        df = pd.concat([self.actual_vacc_df, self.proj_vacc_df])
        return df

    def get_vacc_per_available(self):
        vacc_rates = self.get_vacc_rates()
        populations = [{'age': li['attributes']['age'], 'region': li['attributes']['region'], 'population':li['values']} for li in self.model_params['group_pop'] if li['attributes']['region'] in self.regions]
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

    def get_proj_mobility(self):
        # TODO: implement mobility projections
        return {}

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
        region_pops = {params_list['attributes']['region']: params_list['values'] for params_list in self.model_params['total_pop'] }
        params.update(
            {f"region_pop_{region}": [{'tslices': None, 'attributes': {}, 'values': region_pops[region] }] for region in self.regions}
        )
        return params

    def get_kappas_as_params_using_region_fits(self, engine, region_fit_spec_ids, region_fit_result_ids):
        # TODO: load number of infected from database to compute mobility aware kappas
        # will set TC = 0.0 for the model and scale the "kappa" parameter for each region according to its fitted TC values so the forward sim can be run.
        # This approach does not currently support fitting, since only TC can be fit and there's only one set of TC for the model.
        results_list = []
        for i, tup in enumerate(zip(self.regions, region_fit_spec_ids)):
            region, spec_id = tup
            # load tslices and tcs from the database
            df = pd.read_sql_query(f"select regions, start_date, end_date, tslices, tc, from covid_model.specifications where spec_id = {spec_id}", con=engine, coerce_float=True)
            # make sure region in run spec matches our region
            if json.loads(df['regions'][0])[0] != region:
                ValueError(f'spec_id {spec_id} has region {json.loads(df["regions"][0])[0]} which does not match model\'s {i}th region: {region}')
            tslices = [(df['start_date'][0] + dt.timedelta(days=d) - self.start_date).days for d in [0] + df['tslices'][0]]
            tc = df['tc'][0]
            results_list.append(pd.DataFrame.from_dict({'tslices': tslices, region: tc}).set_index('tslices'))
        # union all the region tslices
        df_tcs = pd.concat(results_list, axis=1)
        tslices = df_tcs.index.drop([0]).to_list()

        # retrieve prevalence data from db if allowing mobility
        if self.model_mobility_mode != 'none':
            results_list = []
            for region, spec_id, result_id in zip(self.regions, region_fit_spec_ids, region_fit_result_ids):
                df = pd.read_sql_query(f"SELECT t, vals FROM covid_model.results_v2 WHERE spec_id = {spec_id} AND result_id = {result_id} order by t", con=engine, coerce_float=True)
                df['infected'] = [sum(itemgetter('I', 'Ih')(json.loads(row))) for row in df['vals']]
                df['pop'] = [sum(json.loads(row).values()) for row in df['vals']]
                df = df.groupby('t').sum(['infected', 'pop'])
                df[region]=df['infected']/df['pop']
                df = df.drop(columns=['infected', 'pop'])
                results_list.append(df)
            prev_df = pd.concat(results_list, axis=1)

        # compute kappa parameter for each region and apply to model parameters
        if self.model_mobility_mode == 'none':
            self.set_tc(tslices, [0.0] * (len(tslices) + 1))
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
            self.set_tc(combined_tslices[1:], [0.0] * len(combined_tslices))
            params = {'kappa_pa': [{'tslices': combined_tslices[1:], 'attributes': {'region': region}, 'values': kappas[:,j].tolist()} for j, region in enumerate(self.regions)]}
        return params
