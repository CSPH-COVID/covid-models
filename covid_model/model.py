### Python Standard Library ###
import json
import math
import datetime as dt
import copy
from operator import itemgetter
import itertools
from collections import OrderedDict, defaultdict
import logging
### Third Party Imports ###
import numpy as np
import pandas as pd
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import scipy.integrate as spi
import scipy.sparse as spsp
from sqlalchemy import func
from sqlalchemy.orm import Session
from sortedcontainers import SortedDict
### Local Imports ###
from covid_model.db import get_sqa_table, db_engine
from covid_model.ode_flow_terms import ConstantODEFlowTerm, ODEFlowTerm
from covid_model.data_imports import ExternalVacc, ExternalHospsEMR, ExternalHospsCOPHS, get_region_mobility_from_db
from covid_model.utils import get_params, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel:
    ####################################################################################################################
    ### Setup
    def log_and_raise(self, ermsg, ErrorType):
        logger.exception(f"{str(self.tags)}" + ermsg)
        raise ErrorType(ermsg)


    ####################################################################################################################
    ### Initialization and Updating
    def __init__(self, engine=None, base_model=None, update_derived_properties=True, base_spec_id=None, **margs):
        # margs can be any model property
        self.recently_updated_properties = []

        # basic model data
        self.__attrs = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                             'age': ['0-19', '20-39', '40-64', '65+'],
                             'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                             'variant': ['none', 'wildtype', 'alpha', 'delta', 'omicron', 'ba2', 'ba2121', 'ba45'],
                             'immun': ['none', 'weak', 'strong'],
                             'region': ['co']})

        self.tags = {}

        self.__start_date = dt.datetime.strptime("2020-01-01", "%Y-%m-%d").date()
        self.__end_date = dt.datetime.strptime("2023-01-01", "%Y-%m-%d").date()

        self.__tc = {}
        self.tc_t_prev_lookup = []
        self.tc_cov = None

        self.solution_y = None

        self.compartments_as_index = None
        self.Ih_compartments = None
        self.compartments = None
        self.cmpt_idx_lookup = None
        self.param_compartments = None

        # model data
        self.__params_defs = None
        self.__region_defs = None
        self.regions = self.attrs['region']
        self.__vacc_proj_params = None
        self.__mobility_mode = None
        self.actual_mobility = None
        self.mobility_proj_params = None
        self.actual_vacc_df = None
        self.proj_vacc_df = None
        self.hosps = None
        self.__hosp_reporting_frac = None

        self.base_spec_id = None
        self.spec_id = None
        self.region_fit_spec_ids = None
        self.region_fit_result_ids = None

        # ode data
        self.t_prev_lookup = None
        self.terms = None
        self.params_by_t = {'all': {}}
        self.__y0_dict = None
        self.flows_string = '(' + ','.join(self.attr_names) + ')'

        self.linear_matrix = None
        self.nonlinear_matrices = None
        self.constant_vector = None
        self.max_step_size = np.inf

        if base_model is not None and base_spec_id is not None:
            self.log_and_raise("Cannot pass both a base_model and base_spec_id", ValueError)

        # if there is a base model, take all its properties
        if base_model is not None:
            logger.debug(f"{str(self.tags)} Copying from base model")
            for key, val in vars(base_model).items():
                setattr(self, key, copy.deepcopy(val))

        # if a base_spec_id provided, load from the database
        if base_spec_id is not None:
            logger.debug(f"{str(self.tags)} Copying base specifications")
            self.base_spec_id = base_spec_id
            self.read_from_base_spec_id(engine)

        # update any attributes with items in **margs
        if len(margs.keys()) > 0:
            logger.debug(f"{str(self.tags)} Applying model arguments")
        for key, val in margs.items():
            setattr(self, key, copy.deepcopy(val))
            self.recently_updated_properties.append(key)

        if update_derived_properties:
            self.update(engine)

    # some properties are derived / computed / constructed from other properties. If the non-derived
    # properties are updated, the derived properties may need recomputing
    def update(self, engine):
        if any([p in self.recently_updated_properties for p in ['regions', 'region_defs']]):
            logger.debug(f"{str(self.tags)} Updating actual vaccines")
            self.set_actual_vacc(engine)

        if any([p in self.recently_updated_properties for p in ['end_date', 'vacc_proj_params']]) and self.vacc_proj_params is not None:
            logger.debug(f"{str(self.tags)} Updating Projected Vaccines")
            self.set_proj_vacc()

        if any([p in self.recently_updated_properties for p in ['end_date', 'mobility_mode']]) and self.mobility_mode is not None:
            logger.debug(f"{str(self.tags)} Updating Actual Mobility")
            self.set_actual_mobility(engine)

        if any([p in self.recently_updated_properties for p in ['end_date', 'mobility_mode', 'mobility_proj_params']]) and self.mobility_mode is not None:
            logger.debug(f"{str(self.tags)} Updating Projected Mobility")
            self.set_proj_mobility()

        if any([p in self.recently_updated_properties for p in ['end_date', 'model_mobility_mode', 'mobility_proj_params']]):
            if self.mobility_mode is not None and self.mobility_mode != "none":
                logger.debug(f"{str(self.tags)} Getting Mobility As Parameters")
                self.params_defs.extend(self.get_mobility_as_params())

        #if any([p in self.recently_updated_properties for p in ['region_fit_spec_ids', 'region_fit_result_ids']]):
            #logger.debug(f"{str(self.tags)} Getting kappas as parameters using region fits")
            #self.params_defs.extend(self.get_kappas_as_params_using_region_fits(engine))

        if any([p in self.recently_updated_properties for p in ['start_date', 'regions', 'region_defs', 'hosp_reporting_frac']]):
            logger.debug(f"{str(self.tags)} Setting Hospitalizations")
            self.set_hosp(engine)

        self.recently_updated_properties = []

    ####################################################################################################################
    ### Functions to Update Derived Properites and Retrieve Data

    def set_actual_vacc(self, engine=None, actual_vacc_df=None):
        logger.info(f"{str(self.tags)} Retrieving vaccinations data")
        if engine is None:
            engine = db_engine()
        logger.debug(f"{str(self.tags)} getting vaccines from db")
        actual_vacc_df_list = []
        for region in self.regions:
            county_ids = self.region_defs[region]['counties_fips']
            actual_vacc_df_list.append(ExternalVacc(engine).fetch(county_ids=county_ids).assign(region=region).set_index('region', append=True).reorder_levels(['measure_date', 'region', 'age']))
        self.actual_vacc_df = pd.concat(actual_vacc_df_list)
        self.actual_vacc_df.index.set_names('date', level=0, inplace=True)
        logger.debug(f"{str(self.tags)} Vaccinations span from {self.actual_vacc_df.index.get_level_values(0).min()} to {self.actual_vacc_df.index.get_level_values(0).max()}")

    def set_proj_vacc(self):
        logger.info(f"{str(self.tags)} Constructing vaccination projections")
        proj_lookback = self.vacc_proj_params['lookback']
        proj_fixed_rates = self.vacc_proj_params['fixed_rates']
        max_cumu = self.vacc_proj_params['max_cumu']
        max_rate_per_remaining = self.vacc_proj_params['max_rate_per_remaining']
        realloc_priority = self.vacc_proj_params['realloc_priority']

        shots = list(self.actual_vacc_df.columns)
        region_df = pd.DataFrame({'region': self.regions})

        # add projections
        proj_from_date = self.actual_vacc_df.index.get_level_values('date').max() + dt.timedelta(days=1)
        proj_to_date = self.end_date
        if proj_to_date >= proj_from_date:
            proj_date_range = pd.date_range(proj_from_date, proj_to_date).date
            # project daily vaccination rates based on the last {proj_lookback} days of data
            projected_rates = self.actual_vacc_df[self.actual_vacc_df.index.get_level_values(0) >= proj_from_date-dt.timedelta(days=proj_lookback)].groupby(['region', 'age']).sum()/proj_lookback
            # override rates using fixed values from proj_fixed_rates, when present
            if proj_fixed_rates:
                proj_fixed_rates_df = pd.DataFrame(proj_fixed_rates).rename_axis(index='age').reset_index().merge(region_df,how='cross').set_index(['region', 'age'])
                for shot in shots:
                    # Note: currently treats all regions the same. Need to change if finer control desired
                    projected_rates[shot] = proj_fixed_rates_df[shot]
            # build projections
            projections = pd.concat({d: projected_rates for d in proj_date_range}).rename_axis(index=['date', 'region', 'age'])

            # reduce rates to prevent cumulative vaccination from exceeding max_cumu
            if max_cumu:
                cumu_vacc = self.actual_vacc_df.groupby(['region', 'age']).sum()
                groups = realloc_priority if realloc_priority else projections.groupby(['region','age']).sum().index
                # self.params_by_t hasn't necessarily been built yet, so use a workaround
                populations = pd.DataFrame([{'region': param_dict['attrs']['region'], 'age': param_dict['attrs']['age'], 'population': list(param_dict['vals'].values())[0]} for param_dict in self.params_defs if param_dict['param'] == 'region_age_pop' and param_dict['attrs']['region'] in self.regions])

                for d in projections.index.unique('date'):
                    this_max_cumu = get_params(max_cumu.copy(), d)

                    # Note: currently treats all regions the same. Need to change if finer control desired
                    max_cumu_df = pd.DataFrame(this_max_cumu).rename_axis(index='age').reset_index().merge(region_df, how='cross').set_index(['region', 'age']).sort_index()
                    max_cumu_df = max_cumu_df.mul(pd.DataFrame(populations).set_index(['region', 'age'])['population'], axis=0)
                    for i in range(len(groups)):
                        group = groups[i]
                        key = tuple([d] + list(group))
                        current_rate = projections.loc[key]
                        max_rate = max_rate_per_remaining * (max_cumu_df.loc[group] - cumu_vacc.loc[group])
                        excess_rate = (projections.loc[key] - max_rate).clip(lower=0)
                        projections.loc[key] -= excess_rate
                        # if a reallocate_order is provided, reallocate excess rate to other groups
                        if i < len(groups) - 1 and realloc_priority is not None:
                            projections.loc[tuple([d] + list(groups[i + 1]))] += excess_rate

                    cumu_vacc += projections.loc[d]

            self.proj_vacc_df = projections
            logger.debug(f"{str(self.tags)} projected vaccinations span from {self.proj_vacc_df.index.get_level_values(0).min()} to {self.proj_vacc_df.index.get_level_values(0).max()}")
        else:
            self.proj_vacc_df = None
            logger.info(f"{str(self.tags)} No vaccine projections necessary")

    def set_actual_mobility(self, engine=None):
        logger.info(f"{str(self.tags)} Retrieving mobility data")
        if engine is None:
            engine = db_engine()
        regions_lookup = pd.DataFrame.from_dict({'county_id': [fips for region in self.regions for fips in self.region_defs[region]['counties_fips']],
                                                 'region': [region for region in self.regions for fips in self.region_defs[region]['counties_fips']]})
        # get mobility and add regions to it
        df = get_region_mobility_from_db(engine, county_ids=regions_lookup['county_id'].to_list()).reset_index('measure_date').rename(columns={'measure_date': 'date'})\
            .join(regions_lookup.rename(columns={'county_id': 'origin_county_id', 'region': 'from_region'}).set_index('origin_county_id'), on='origin_county_id')\
            .join(regions_lookup.rename(columns={'county_id': 'destination_county_id', 'region': 'to_region'}).set_index('destination_county_id'), on='destination_county_id')

        # aggregate by t and origin / dest regions
        df = df.drop(columns=['origin_county_id', 'destination_county_id']) \
            .groupby(['date', 'from_region', 'to_region']) \
            .aggregate(total_hrs=('total_dwell_duration_hrs', 'sum'))

        self.actual_mobility = df
        logger.debug(f"{str(self.tags)} mobility spans from {self.actual_mobility.index.get_level_values(0).min()} to {self.actual_mobility.index.get_level_values(0).max()}")

    def set_proj_mobility(self):
        # TODO: implement mobility projections
        logger.info(f"{str(self.tags)} Constructing Mobility projections")
        #self.proj_mobility = pd.DataFrame(columns = self.actual_mobility.columns)
        self.proj_mobility = None
        logger.warning(f"{str(self.tags)} mobility projections not yet implemented")

    def get_mobility_as_params(self):
        logger.info(f"{str(self.tags)} Converting mobility data into parameters")
        # compute fraction in and fraction from
        mobility = pd.concat([self.actual_mobility, self.proj_mobility]) if self.proj_mobility is not None else self.actual_mobility
        mobility = mobility\
            .join(mobility.groupby(['date', 'from_region']).aggregate(total_from_hrs=('total_hrs', 'sum')))\
            .join(mobility.groupby(['date', 'to_region']).aggregate(total_to_hrs=('total_hrs', 'sum')))\
            .reorder_levels([1, 2, 0]).sort_index()
        mobility['frac_of_from'] = mobility['total_hrs'] / mobility['total_from_hrs']  # fraction of "from"'s hours spent in "to"
        mobility['frac_of_to'] = mobility['total_hrs'] / mobility['total_to_hrs']   # fraction of hours spent in "to" that are from "from".

        # convert index to string, so they can be serialized more easily.
        mobility.reset_index('date', inplace=True)
        mobility['date'] = [dt.datetime.strftime(d, '%Y-%m-%d') for d in mobility['date']]
        mobility.set_index('date', append=True, inplace=True)

        params = []
        if self.mobility_mode == 'population_attached':
            # self join to compute the dot product over all intermediate regions
            S_in_region = mobility[['frac_of_from']].reset_index(['to_region', 'from_region'])\
                .rename(columns={'to_region': 'intermediate_region', 'from_region': 'susceptible_region', 'frac_of_from': 'frac_of_susceptible_region' })\
                .set_index(['susceptible_region', 'intermediate_region'], append=True)
            I_in_region = mobility[['frac_of_to']].reset_index(['to_region', 'from_region'])\
                .rename(columns={'to_region': 'intermediate_region', 'from_region': 'infectious_region', 'frac_of_to': 'frac_from_infectious_region' }) \
                .set_index(['infectious_region', 'intermediate_region'], append=True)
            I_S_contact = S_in_region.join(I_in_region)\
                .reorder_levels(['date', 'susceptible_region', 'infectious_region', 'intermediate_region'])\
                .assign(I_S_contact = lambda x: x['frac_of_susceptible_region'] * x['frac_from_infectious_region'])\
                .groupby(['date', 'susceptible_region', 'infectious_region'])\
                .aggregate(I_S_contact = ('I_S_contact', 'sum'))\
                .reorder_levels(['susceptible_region', 'infectious_region', 'date']).sort_index()

            for S_region in I_S_contact.index.get_level_values('susceptible_region').unique():
                for I_region in I_S_contact.index.get_level_values('infectious_region').unique():
                    params.extend([{'param': f'mob_{S_region}_exposure_from_{I_region}',  'attrs': {}, 'vals': I_S_contact.loc[(S_region, I_region)]['I_S_contact'].to_dict()}])

        elif self.mobility_mode == 'location_attached':
            for from_region in mobility.index.get_level_values('from_region').unique():
                for to_region in mobility.index.get_level_values('to_region').unique():
                    params.extend([{'param': f'mob_{from_region}_frac_in_{to_region}',  'attrs': {}, 'vals': mobility.loc[(from_region, to_region)]['frac_of_from'].to_dict()}])
                    params.extend([{'param': f'mob_{to_region}_frac_from_{from_region}', 'attrs': {}, 'vals': mobility.loc[(from_region, to_region)]['frac_of_to'].to_dict()}])

        return params

    def get_kappas_as_params_using_region_fits(self, engine=None):
        if engine is None:
            engine = db_engine()
        # will set TC = 0.0 for the model and scale the "kappa" parameter for each region according to its fitted TC values so the forward sim can be run.
        # This approach does not currently support fitting, since only TC can be fit and there's only one set of TC for the model.
        results_list = []
        for i, (region, spec_id) in enumerate(zip(self.regions, self.region_fit_spec_ids)):
            # load tslices and tcs from the database
            df = pd.read_sql_query(f"select regions, start_date, end_date, tslices, tc, from covid_model.specifications where spec_id = {spec_id}", con=engine, coerce_float=True).loc[0]
            # make sure region in run spec matches our region
            if json.loads(df['regions'])[0] != region:
                self.log_and_raise(f'spec_id {spec_id} has region {json.loads(df["regions"])[0]} which does not match model\'s {i}th region: {region}', ValueError)
            tslices = [(df['start_date'] + dt.timedelta(days=d) - self.start_date).days for d in [0] + df['tslices']]
            tc = df['tc']
            results_list.append(pd.DataFrame.from_dict({'tslices': tslices, region: tc}).set_index('tslices'))
        # union all the region tslices
        df_tcs = pd.concat(results_list, axis=1)
        tslices = df_tcs.index.to_list()

        # retrieve prevalence data from db if allowing mobility
        prev_df = None
        if self.mobility_mode is not None:
            results_list = []
            for region, spec_id, result_id in zip(self.regions, self.region_fit_spec_ids, self.region_fit_result_ids):
                df = pd.read_sql_query(f"SELECT date, vals FROM covid_model.results_v2 WHERE spec_id = {spec_id} AND result_id = {result_id} order by t", con=engine, coerce_float=True)
                df['infected'] = [sum(itemgetter('I', 'Ih')(json.loads(row))) for row in df['vals']]
                df['pop'] = [sum(json.loads(row).values()) for row in df['vals']]
                df = df.groupby('date').sum(['infected', 'pop'])
                df[region]=df['infected']/df['pop']
                df = df.drop(columns=['infected', 'pop'])
                results_list.append(df)
            prev_df = pd.concat(results_list, axis=1)
            prev_df['t'] = [self.date_to_t(d) for d in prev_df.index]

        # compute kappa parameter for each region and apply to model parameters
        params = {}
        if self.mobility_mode is None:
            self.__tc = [0.0] * (len(tslices) + 1)
            params = [{'param': 'kappa', 'attrs': {'region':region}, 'vals': dict(zip(tslices, [(1-tc) for tc in df_tcs[region]]))} for region in df_tcs.columns]
        elif self.mobility_mode == 'location_attached':
            # TODO: Implement
            params = []
        elif self.mobility_mode == 'population_attached':
            # average the mobility over each tc window from the region tc fits
            zero_mat = np.zeros_like(list(self.actual_mobility.values())[0]['dwell'])
            mobility_ave_for_tc_interval = {t: {'dwell': zero_mat.copy(), 'dwell_rownorm': zero_mat.copy(), 'dwell_colnorm': zero_mat.copy()} for t in df_tcs.index}
            for t0, t1 in zip(list(self.actual_mobility.keys())[:-1], list(self.actual_mobility.keys())[1:]):
                if t0 > max(df_tcs.index):
                    break
                t_tc0 = [t for t in df_tcs.index if t <= t0][-1]
                t_tc1 = [t for t in df_tcs.index if t_tc0 < t][0]
                t_tc2 = [t for t in df_tcs.index if t_tc1 < t][0] if t_tc1 != max(df_tcs.index) else None
                days_in_t_tc0 = min(t_tc1, t1) - max(t_tc0, t0)
                days_in_t_tc1 = t1 - t0 - days_in_t_tc0
                for key in ['dwell', 'dwell_rownorm', 'dwell_colnorm']:
                    mobility_ave_for_tc_interval[t_tc0][key] += np.array(self.actual_mobility[t0][key]) * days_in_t_tc0 / (t_tc1-t_tc0)
                    mobility_ave_for_tc_interval[t_tc1][key] += np.array(self.actual_mobility[t0][key]) * [days_in_t_tc1 / (t_tc2-t_tc1) if t_tc2 is not None else 1][0]
            # average the prevalence over each tc window from the region tc fits
            prev_t_rounded_down_to_mob_tslices = pd.cut(prev_df['t'], list(df_tcs.index) + [self.tend], right=False, retbins=False, labels=df_tcs.index)
            prev_df_binned = prev_df.groupby(prev_t_rounded_down_to_mob_tslices).mean().drop(columns=['t'])

            kappas = np.zeros(shape=df_tcs.shape)
            for i, t in enumerate(df_tcs.index):
                tc = df_tcs.loc[t].to_numpy()
                prev = prev_df_binned.loc[t]
                mobs = mobility_ave_for_tc_interval[t]
                kappas[i, :] = (1-tc) * prev / np.linalg.multi_dot([mobs['dwell_rownorm'], np.transpose(mobs['dwell_colnorm']), np.transpose(prev)])
            np.nan_to_num(kappas, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.update_tc([0.0] * len(self.__tc))
            dates = [self.t_to_date(d) for d in df_tcs.index]
            params = [{'param': 'kappa_pa', 'attrs': {'region': region}, 'vals': dict(zip(dates, kappas[:, j].tolist())), 'desc': ''} for j, region in enumerate(self.regions)]
        return params

    def hosp_reporting_frac_by_t(self):
        # currently assigns the same hrf to all regions.
        hrf = pd.DataFrame(index=self.daterange)
        hrf['hosp_reporting_frac'] = np.nan
        early_dates = []
        for date, val in self.hosp_reporting_frac.items():
            date = date if isinstance(date, dt.date) else dt.datetime.strptime(date, "%Y-%m-%d").date()
            if date in hrf.index:
                hrf.loc[date]['hosp_reporting_frac'] = val
            elif date <= hrf.index[0]:
                early_dates.append(date)
        if len(early_dates) > 0:
            # among all the dates before the start date, take the latest one
            hrf.iloc[0] = self.hosp_reporting_frac[dt.datetime.strftime(max(early_dates), "%Y-%m-%d")]
        hrf = hrf.ffill()
        hrf.index.name = 'date'
        hrf = pd.concat([hrf.assign(region=r) for r in self.attrs['region']]).set_index('region', append=True).reorder_levels([1,0])
        return hrf

    # pulls hospitalizations for only the first region, since the model only fits one region at a time
    def set_hosp(self, engine=None):
        if engine is None:
            engine = db_engine()
        logger.info(f"{str(self.tags)} Retrieving hospitalizations data")
        # makes sure we pull from EMResource if region is CO
        #county_ids = self.region_defs[self.regions[0]]['counties_fips'] if self.regions[0] != 'co' else None
        regions_lookup = pd.DataFrame.from_dict({'county_id': [fips for region in self.regions for fips in self.region_defs[region]['counties_fips']],
                                                 'region': [region for region in self.regions for fips in self.region_defs[region]['counties_fips']]})
        # TODO: fix for CO
        if self.regions != ['co']:
            hosps = ExternalHospsCOPHS(engine).fetch(county_ids=regions_lookup['county_id'].to_list())\
                .join(regions_lookup.set_index('county_id'), on='county_id')\
                .groupby(['measure_date', 'region'])\
                .aggregate(observed=('observed_hosp', 'sum'))\
                .reset_index('measure_date')\
                .rename(columns={'measure_date': 'date'})\
                .set_index('date', append=True).sort_index()
        else:
            hosps = ExternalHospsEMR(engine).fetch()\
                .rename(columns={'currently_hospitalized': 'observed'})\
                .assign(region='co')\
                .reset_index('measure_date')\
                .rename(columns={'measure_date': 'date'})\
                .set_index(['region', 'date']).sort_index()
        # fill in the beginning if necessary, or truncate if necessary
        hosps = hosps.reindex(pd.MultiIndex.from_product([self.regions, pd.date_range(self.start_date, max(hosps.index.get_level_values(1))).date], names=['region', 'date']), fill_value=0)

        hosps = hosps.join(self.hosp_reporting_frac_by_t())
        hosps['estimated_actual'] = hosps['observed'] / hosps['hosp_reporting_frac']
        self.hosps = hosps


    ####################################################################################################################
    ### Properites

    ### Date / Time. Updating start date and end date updates other date/time attributes also
    @property
    def start_date(self):
        return self.__start_date

    @start_date.setter
    def start_date(self, value):
        start_date = value if isinstance(value, dt.date) else dt.datetime.strptime(value, "%Y-%m-%d").date()
        # shift tc to the right if start date is earlier; shift left and possibly truncate TC if start date is later
        tshift = (start_date-self.start_date).days
        self.__start_date = start_date
        self.__tend = (self.end_date - self.start_date).days
        self.__trange = range(self.tstart, self.tend + 1)
        self.__daterange = pd.date_range(self.start_date, end=self.end_date).date
        self.__tc = {(t - tshift): tc for t, tc in self.__tc.items() if t >= tshift}
        if 0 not in self.__tc.keys() and any(tc_t < 0 for tc_t in self.__tc.keys()):
            self.__tc[0] = self.__tc[max(tc_t for tc_t in self.__tc.keys() if tc_t < 0)]
        if any(tc_t >= self.tstart for tc_t in self.__tc.keys()):
            self.tc_t_prev_lookup = [max(t for t in self.__tc.keys() if t <= t_int) for t_int in self.trange]
        self.recently_updated_properties.append('start_date')

    @property
    def end_date(self):
        return self.__end_date

    @end_date.setter
    def end_date(self, value):
        end_date = value if isinstance(value, dt.date) else dt.datetime.strptime(value, "%Y-%m-%d").date()
        # truncate tc if end date shortens
        self.__end_date = end_date
        self.__tend = (self.end_date - self.start_date).days
        self.__trange = range(self.tstart, self.tend + 1)
        self.__daterange = pd.date_range(self.start_date, end=self.end_date).date
        self.__tc = {t: tc for t, tc in self.__tc.items() if t <= self.tend}
        if any(tc_t <= self.tend for tc_t in self.__tc.keys()):
            self.tc_t_prev_lookup = [max(t for t in self.__tc.keys() if t <= t_int) for t_int in self.trange]
        self.recently_updated_properties.append('end_date')

    @property
    def tstart(self):
        return 0

    @property
    def tend(self):
        return self.__tend

    @tend.setter
    def tend(self, value):
        self.__tend = value
        self.__end_date = self.start_date + dt.timedelta(days=value)
        self.__trange = range(self.tstart, self.tend + 1)
        self.__daterange = pd.date_range(self.start_date, end=self.end_date).date
        self.recently_updated_properties.append('end_date')

    @property
    def trange(self):
        return self.__trange

    @property
    def daterange(self):
        return self.__daterange

    @property
    def attr_names(self):
        return list(self.attrs.keys())

    @property
    def param_attr_names(self):
        return self.attr_names[1:]

    ### attributes and regions
    def update_compartments(self):
        self.compartments_as_index = pd.MultiIndex.from_product(self.attrs.values(), names=self.attr_names)
        self.Ih_compartments = self.compartments_as_index.get_level_values(0) == "Ih"
        self.compartments = list(self.compartments_as_index)
        self.cmpt_idx_lookup = pd.Series(index=self.compartments_as_index, data=range(len(self.compartments_as_index))).to_dict()
        self.param_compartments = list(set(tuple(attr_val for attr_val, attr_name in zip(cmpt, self.attr_names) if attr_name in self.param_attr_names) for cmpt in self.compartments))


    @property
    def regions(self):
        return self.__regions

    @regions.setter
    def regions(self, value: list):
        self.__regions = value
        # if regions changes, update the compartment attributes, and everything derived from that as well.
        self.__attrs['region'] = value
        self.update_compartments()
        self.recently_updated_properties.append('regions')

    @property
    def attrs(self):
        return self.__attrs

    @attrs.setter
    def attrs(self, value: OrderedDict):
        self.__attrs = value
        self.__regions = value['region']
        self.update_compartments()
        self.recently_updated_properties.append('regions')

    ### things which are dictionaries but which may be given as a path to a json file
    @property
    def params_defs(self):
        return self.__params_defs

    @params_defs.setter
    def params_defs(self, value):
        self.__params_defs = value if isinstance(value, list) else json.load(open(value))
        self.recently_updated_properties.append('params_defs')

    @property
    def vacc_proj_params(self):
        return self.__vacc_proj_params

    @vacc_proj_params.setter
    def vacc_proj_params(self, value):
        self.__vacc_proj_params = value if isinstance(value, dict) else json.load(open(value))
        self.recently_updated_properties.append('vacc_proj_params')

    @property
    def region_defs(self):
        return self.__region_defs

    @region_defs.setter
    def region_defs(self, value):
        self.__region_defs = value if isinstance(value, dict) else json.load(open(value))
        self.recently_updated_properties.append('region_defs')

    ### Set mobility mode as None if "none"
    @property
    def mobility_mode(self):
        return self.__mobility_mode

    @mobility_mode.setter
    def mobility_mode(self, value):
        self.__mobility_mode = value if value != 'none' else None
        self.recently_updated_properties.append('mobility_mode')

    @property
    def hosp_reporting_frac(self):
        return self.__hosp_reporting_frac if self.__hosp_reporting_frac is not None else {dt.datetime.strftime(self.start_date, "%Y-%m-%d"): 1}

    @hosp_reporting_frac.setter
    def hosp_reporting_frac(self, value):
        self.__hosp_reporting_frac = value
        self.recently_updated_properties.append('hosp_reporting_frac')

    ### Properties that take a little computation to get

    # initial state y0, expressed as a dictionary with non-empty compartments as keys
    @property
    def y0_dict(self):
        if self.__y0_dict is None:
            group_pops = self.get_param_for_attrs_by_t('region_age_pop', attrs={'vacc': 'none', 'variant': 'none', 'immun': 'none'}).loc[0].reset_index().drop(columns=['vacc', 'variant', 'immun'])
            y0d = {('S', row['age'], 'none', 'none', 'none', row['region']): row['region_age_pop'] for i, row in group_pops.iterrows()}
            self.__y0_dict = y0d
            return y0d
        else:
            return self.__y0_dict

    @y0_dict.setter
    def y0_dict(self, val: dict):
        # set everything to zero at first
        y0d = {}
        for cmpt, v in val.items():
            if cmpt in self.compartments:
                y0d[cmpt] = v
            else:
                self.log_and_raise(f'{cmpt} not a valid compartment', ValueError)
        self.__y0_dict = y0d

    @property
    def tc(self):
        return self.__tc


    # return the parameters as nested dictionaries
    @property
    def params_as_dict(self, params=None):
        params_dict = {}
        for cmpt, cmpt_dict in self.params_by_t.items():
            key = f"({','.join(cmpt)})" if isinstance(cmpt, tuple) else cmpt
            params_dict[key] = cmpt_dict
        return params_dict

    @property
    def n_compartments(self):
        return len(self.cmpt_idx_lookup)

    @property
    def solution_ydf(self):
        return pd.concat([self.y_to_series(self.solution_y[t]) for t in self.trange], axis=1, keys=self.trange, names=['t']).transpose()

    @property
    def new_infections(self):
        param_df = self.get_param_for_attrs_by_t('alpha', attrs={}, convert_to_dates=True)
        combined = self.solution_sum_df()[['E']].stack(self.param_attr_names).join(param_df)
        combined['Enew'] = (combined['E'] / combined['alpha'])
        combined = combined.groupby(['date', 'region']).sum().drop(columns=['E', 'alpha'])
        return combined

    @property
    def re_estimates(self):
        param_df = self.get_params_for_attrs_by_t(['gamm', 'alpha'], attrs={}, convert_to_dates=True)
        combined = self.solution_sum_df()[['I', 'A', 'E']].stack(self.param_attr_names).join(param_df)
        combined['infect_duration'] = 1/combined['gamm']
        combined['lagged_infected'] = combined.groupby('region').shift(3)[['I', 'A']].sum(axis=1)
        combined['new_infected'] = combined['E'] / combined['alpha']
        combined = combined.groupby(['date', 'region']).agg({'new_infected':'sum', 'lagged_infected': 'sum', 'infect_duration': 'mean'})
        combined['re'] = combined['new_infected'] * combined['infect_duration'] / combined['lagged_infected']
        return combined[['re']]


    ####################################################################################################################
    ### useful getters

    def date_to_t(self, date):
        if isinstance(date, str):
            return (dt.datetime.strptime(date, "%Y-%m-%d").date() - self.start_date).days
        else:
            return (date - self.start_date).days

    def t_to_date(self, t):
        return self.start_date + dt.timedelta(days=t)

    def get_vacc_rates(self):
        df = pd.concat([self.actual_vacc_df, self.proj_vacc_df])
        return df

    # get the state at time t as a dictionary
    def y_dict(self, t):
        return {cmpt: y for cmpt, y in zip(self.compartments, self.solution_y[t, :])}

    # create a y0 vector with all values as 0, except those designated in y0_dict
    def y0_from_dict(self, y0_dict):
        y0 = [0] * self.n_compartments
        for cmpt, n in y0_dict.items():
            y0[self.cmpt_idx_lookup[cmpt]] = n
        return y0

    # returns list of fips codes for each county in the given region, or every region in this model if not given
    def get_all_county_fips(self, regions=None):
        regions = self.regions if regions is None else regions
        return [county_fips for region in regions for county_fips in self.region_defs[region]['counties_fips']]

    # convert y-array to series with compartment attributes as multiindex
    def y_to_series(self, y):
        return pd.Series(index=self.compartments_as_index, data=y)

    # give the counts for all compartments over time, but group by the compartment attributes listed
    def solution_sum_df(self, group_by_attr_levels=None):
        df = self.solution_ydf
        if group_by_attr_levels is not None:
            df = df.groupby(group_by_attr_levels, axis=1).sum()
        df['date'] = self.daterange
        df = df.set_index('date')
        return df

    # gives hospitalizations, separated by region, as a numpy array where rows are time and columns are region
    def solution_sum_Ih(self, tstart=0, tend=None, regions=None):
        tend = self.tend if tend is None else tend
        regions = self.regions if regions is None else regions

        region_levels = self.compartments_as_index.get_level_values(-1)
        Ih = np.concatenate([self.solution_y[tstart:(tend+1), self.Ih_compartments & (region_levels == region)].sum(axis=1) for region in regions])
        return Ih

    # Get the immunity against a given variant.
    # I.e. if everyone were exposed today, what fraction of people who WOULD be normally infected if the had no immunity, are NOT infected because they are immune?
    # or, if to_hosp=True, then what fraction of people who WOULD be normally hospitalized if they had no immunity, are NOT hospitalized because they are immune?
    def immunity(self, variant='omicron', vacc_only=False, to_hosp=False, age=None):
        group_by_attr_names = self.param_attr_names
        n = self.solution_sum_df(group_by_attr_names).stack(level=group_by_attr_names)
        if age is not None:
            n = n.xs(age, level='age')

        from_attrs = {} if age is None else {'age': age}
        to_attrs = {'variant': variant} if age is None else {'variant': variant, 'age': age}
        if to_hosp:
            from_params = self.get_params_for_attrs_by_t(['immunity', 'severe_immunity', 'hosp', 'mab_prev', 'mab_hosp_adj', 'pax_prev', 'pax_hosp_adj'], attrs=from_attrs)
        else:
            from_params = self.get_params_for_attrs_by_t(['immunity'], attrs=from_attrs)
        from_to_params = self.get_params_for_attrs_by_t(['immune_escape'], from_attrs=from_attrs, to_attrs=to_attrs)
        drop_cols = [name for name in from_to_params.index.names if 'to' in name]
        from_to_params = from_to_params.reset_index(drop_cols).drop(columns=drop_cols)
        from_to_params.index.set_names([name[5:] if 'from_' in name else name for name in from_to_params.index.names], inplace=True)
        params = from_params.join(from_to_params).reset_index('t')
        params['date'] = [self.t_to_date(t) for t in params['t']]
        if age is not None:
            params = params.reset_index('age').drop(columns='age')
        params = params.drop(columns='t').set_index('date', append=True).reorder_levels(n.index.names)

        if vacc_only:
            params.loc[params.index.get_level_values('vacc') == 'none', 'immunity'] = 0
            params.loc[params.index.get_level_values('vacc') == 'none', 'severe_immunity'] = 0

        params['effective_inf_rate'] = 1-params['immunity'] * (1-params['immune_escape'])
        if to_hosp:
            # weights = people who would be hospitalized if noone had any immunity
            weights = n * params['hosp'] * (1-params['mab_prev']-params['pax_prev'] + params['mab_prev']*params['mab_hosp_adj'] + params['pax_prev']*params['pax_hosp_adj'])
            params['effective_hosp_rate'] = params['effective_inf_rate'] * (1-params['severe_immunity'])
            return (weights * (1 - params['effective_hosp_rate'])).groupby('date').sum() / weights.groupby('date').sum()
            #return (n * (1 - params['effective_hosp_rate'])).groupby('date').sum() / n.groupby('date').sum()
        else:
            return (n * (1 - params['effective_inf_rate'])).groupby('date').sum() / n.groupby('date').sum()

    # risk is similar to immunity, but it incorporates prevalence as well. i.e. higher prevalence will lead to higher risk
    def risk(self, variant=None, to_hosp=False, age=None):
        pass


    def modeled_vs_observed_hosps(self):
        df = self.solution_sum_df(['seir', 'region'])['Ih'].stack('region').rename('modeled_actual').to_frame()
        df = df.join(self.hosps)
        df['hosp_reporting_frac'].ffill(inplace=True)
        df['modeled_observed'] = df['modeled_actual'] * df['hosp_reporting_frac']
        df = df.reindex(columns=['observed', 'estimated_actual', 'modeled_actual', 'modeled_observed'])
        df = df.reorder_levels([1,0]).sort_index() # put region first
        return df

    # get a parameter for a given set of attributes and trange. Either specify attrs for compartment parameters, or from_attrs and to_attrs for compartment-pair attributes
    def get_param_for_attrs_by_t(self, param, attrs=None, from_attrs=None, to_attrs=None, convert_to_dates=False):
        # get the keys for the parameters we want
        df_list = []
        if attrs is not None:
            cmpts = self.get_cmpts_matching_attrs(attrs, is_param_cmpts=True)
            for cmpt in cmpts:
                param_key = self.get_param_key_for_param_and_cmpts(param, cmpt=cmpt, is_param_cmpt=True)
                df = pd.DataFrame(index=self.trange, columns=self.param_attr_names, data=[cmpt for t in self.trange]).rename_axis('t').set_index(self.param_attr_names, append=True).assign(**{param: np.nan})
                for t, val in self.params_by_t[param_key][param].items():
                    df[param][t] = val
                df_list.append(df.ffill())
        else:
            from_cmpts = self.get_cmpts_matching_attrs(from_attrs, is_param_cmpts=True)
            for from_cmpt in from_cmpts:
                to_cmpts = self.update_cmpt_tuple_with_attrs(from_cmpt, to_attrs, is_param_cmpt=True)
                for to_cmpt in to_cmpts:
                    param_key = self.get_param_key_for_param_and_cmpts(param, from_cmpt=from_cmpt, to_cmpt=to_cmpt, is_param_cmpt=True)
                    to_from_cols = ["from_" + name for name in self.param_attr_names] + ['to_' + name for name in self.param_attr_names]
                    df = pd.DataFrame(index=self.trange, columns=to_from_cols, data=[from_cmpt + to_cmpt for t in self.trange])\
                        .rename_axis('t').set_index(to_from_cols, append=True).assign(**{param: np.nan})
                    for t, val in self.params_by_t[param_key][param].items():
                        df[param][t] = val
                    df_list.append(df.ffill())
        df = pd.concat(df_list)
        if convert_to_dates:
            df['date'] = [self.t_to_date(t) for t in df.index.get_level_values('t')]
            df = df.reset_index().set_index(['date'] + df.index.names[1:]).drop(columns=['t'])
        return df

    def get_params_for_attrs_by_t(self, params: list, attrs=None, from_attrs=None, to_attrs=None, convert_to_dates=False):
        return pd.concat([self.get_param_for_attrs_by_t(param, attrs, from_attrs, to_attrs, convert_to_dates) for param in params], axis=1)

    # get all terms that refer to flow from one specific compartment to another
    def get_terms_by_cmpt(self, from_cmpt, to_cmpt):
        return [term for term in self.terms if term.from_cmpt_idx == self.cmpt_idx_lookup[from_cmpt] and term.to_cmpt_idx == self.cmpt_idx_lookup[to_cmpt]]

    # get the terms that refer to flow from compartments with a set of attributes to compartments with another set of attributes
    def get_terms_by_attr(self, from_attrs, to_attrs):
        idx = [i for i, term in enumerate(self.terms) if self.does_cmpt_have_attrs(self.compartments[term.from_cmpt_idx], from_attrs) and self.does_cmpt_have_attrs(self.compartments[term.to_cmpt_idx], to_attrs)]
        return [self.terms[i] for i in idx]

    # create a json string capturing all of the ode terms: nonlinear, linear, and constant
    def ode_terms_as_json(self, compact=False):
        if compact:
            cm = ", ".join([f'[{i},{c}]' for i, c in enumerate(self.compartments)])
            cv = [[t, spsp.csr_array(vec)] for t, vec in self.constant_vector.items() if any(vec != 0)]
            cv = {t: ' ,'.join([f'({idx},{val:.2e})' for idx, val in zip(m.nonzero()[1].tolist(), m[m.nonzero()].tolist())]) for t, m in cv}
            lm = {t: ' ,'.join([f'({idx1},{idx2},{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[0].tolist(), m.nonzero()[1].tolist(), m[m.nonzero()].A[0].tolist())]) for t, m in self.linear_matrix.items() if len(m.nonzero()[0]) > 0}
            nl = {t: {f'({",".join([f"{k}" for k in keys])})': ', '.join([f'({idx1},{idx2},{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[0].tolist(), m.nonzero()[1].tolist(), m[m.nonzero()].A[0].tolist()) if val != 0]) for keys, m in mat_dict.items()} for t, mat_dict in self.nonlinear_matrices.items() if len(mat_dict) > 0}
            return json.dumps({"compartments": cm, "constant_vector": cv, "linear_matrix": lm, "nonlinear_matrices": nl}, indent=2)
        else:
            def fcm(i):
                return f'{",".join(self.compartments[i])}'

            cv = [[t, spsp.csr_array(vec)] for t, vec in self.constant_vector.items() if any(vec != 0)]
            cv = {t: {fcm(idx): f'{val:.2e}' for idx, val in zip(m.nonzero()[1].tolist(), m[m.nonzero()].tolist())} for t, m in cv}
            lm = {t: {f'({fcm(idx1)};{fcm(idx2)}': f'{val:.2e}' for idx1, idx2, val in zip(m.nonzero()[1].tolist(), m.nonzero()[0].tolist(), m[m.nonzero()].A[0].tolist())} for t, m in self.linear_matrix.items() if len(m.nonzero()[0]) > 0}
            nl = {t: {f'({";".join([f"{fcm(k)}" for k in keys])})': {f'({fcm(idx1)};{fcm(idx2)})': f'{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[1].tolist(), m.nonzero()[0].tolist(), m[m.nonzero()].A[ 0].tolist()) if val != 0} for keys, m in mat_dict.items()} for t, mat_dict in self.nonlinear_matrices.items() if len(mat_dict) > 0}
            return json.dumps({"constant_vector": cv, "linear_matrix": lm, "nonlinear_matrices": nl}, indent=2)



    ####################################################################################################################
    ### ODE related functions

    # check if a cmpt matches a dictionary of attributes
    def does_cmpt_have_attrs(self, cmpt, attrs, is_param_cmpts=False):
        return all(
            cmpt[self.param_attr_names.index(attr_name) if is_param_cmpts else list(self.attr_names).index(attr_name)]
            in ([attr_val] if isinstance(attr_val, str) else attr_val)
            for attr_name, attr_val in attrs.items())

    # return compartments that match a dictionary of attributes
    def get_cmpts_matching_attrs(self, attrs, is_param_cmpts=False):
        # it's actually faster to construct compartments than filtering them.
        # if compartments become no longer the cartesian product of attributes, then this will break
        attrs = {key: val if isinstance(val, list) else [val] for key, val in attrs.items()}  # convert strings to lists of length 1
        new_attrs = copy.deepcopy(self.attrs)
        if is_param_cmpts:
            _ = [new_attrs.pop(key) for key in [name for name in self.attr_names if name not in self.param_attr_names]]
        new_attrs.update(attrs)
        return list(itertools.product(*list(new_attrs.values())))
        #return [cmpt for cmpt in (self.param_compartments if is_param_cmpts else self.compartments) if self.does_cmpt_have_attrs(cmpt, attrs, is_param_cmpts)]

    # given a compartment tuple, update the corresponding elements with items from attrs dict.
    # if one or more values in attrs dict is a list, then create all combinations of matching compartments
    def update_cmpt_tuple_with_attrs(self, cmpt: tuple, attrs: dict, is_param_cmpt=False):
        cmpt = list(cmpt)
        cmpt_attrs = {attr_name: attr_val for attr_name, attr_val in zip(self.param_attr_names if is_param_cmpt else self.attrs, cmpt)}
        for attr_name, new_attr_val in attrs.items():
            cmpt_attrs[attr_name] = new_attr_val
        return self.get_cmpts_matching_attrs(cmpt_attrs, is_param_cmpt)

    ####################################################################################################################
    ### Prepping and Running

    def get_vacc_per_available(self):
        vacc_rates = self.get_vacc_rates()
        missing_dates = pd.DataFrame({'date': [d for d in self.daterange if d < min(vacc_rates.index.get_level_values('date'))]})
        missing_shots = missing_dates.merge(pd.DataFrame(index=vacc_rates.reset_index('date').index.unique(), columns=vacc_rates.columns).fillna(0).reset_index(), 'cross').set_index(['date', 'region', 'age'])
        vacc_rates = pd.concat([missing_shots, vacc_rates])
        vacc_rates['t'] = [self.date_to_t(d) for d in vacc_rates.index.get_level_values('date')]
        vacc_rates = vacc_rates.set_index('t', append=True)
        populations = self.get_param_for_attrs_by_t('region_age_pop', attrs={'vacc': 'none', 'variant': 'none', 'immun': 'none'}).reset_index([an for an in self.param_attr_names if an not in ['region', 'age']])[['region_age_pop']]
        vacc_rates_ts = vacc_rates.index.get_level_values('t').unique()
        populations = populations.iloc[[t in vacc_rates_ts for t in populations.index.get_level_values('t')]].reorder_levels(['region', 'age', 't'])
        populations.rename(columns={'region_age_pop': 'population'}, inplace=True)
        cumu_vacc = vacc_rates.groupby(['region', 'age']).cumsum()
        cumu_vacc_final_shot = cumu_vacc - cumu_vacc.shift(-1, axis=1).fillna(0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.join(populations)
        # vaccinations eventually overtake population (data issue) which would make 'none' < 0 so clip at 0
        cumu_vacc_final_shot['none'] = (cumu_vacc_final_shot['population'] * 2 - cumu_vacc_final_shot.sum(axis=1)).clip(lower=0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.drop(columns='population')
        cumu_vacc_final_shot = cumu_vacc_final_shot.reindex(columns=['none', 'shot1', 'shot2', 'shot3'])

        available_for_vacc = cumu_vacc_final_shot.shift(1, axis=1).drop(columns='none')
        vacc_per_available = (vacc_rates / available_for_vacc).fillna(0).replace(np.inf, 0).reorder_levels(['t', 'date', 'region', 'age']).sort_index()
        # because vaccinations exceed the population, we can get rates greater than 1. To prevent compartments have negative people, we have to cap the rate at 1
        vacc_per_available = vacc_per_available.clip(upper=1)
        return vacc_per_available

    def set_param_by_t(self, param, vals: dict, mults: dict,  cmpt=None, from_cmpt=None, to_cmpt=None):
        param_key = cmpt if cmpt is not None else (from_cmpt, to_cmpt)
        # cmpts are added to params greedily to reduce space allocation
        if param_key not in self.params_by_t.keys():
            self.params_by_t[param_key] = {}
        if vals is not None:
            # always overwrite all values if vals specified
            self.params_by_t[param_key][param] = SortedDict()
            for d, val in vals.items():
                t = max(self.date_to_t(d), self.tstart)
                if t > self.tend:
                    continue
                self.params_by_t[param_key][param][t] = val
        if mults is not None:
            # copy vals from more general key (e.g. "all") if param not present in this key
            if param not in self.params_by_t[param_key]:
                more_general_key = self.get_param_key_for_param_and_cmpts(param, cmpt, from_cmpt, to_cmpt, is_param_cmpt=True)
                self.params_by_t[param_key][param] = copy.deepcopy(self.params_by_t[more_general_key][param])
            # add in tslices which are missing
            for d in sorted(list(mults.keys())):
                t = max(self.date_to_t(d), self.tstart)
                if t > self.tend:
                    continue
                if not self.params_by_t[param_key][param].__contains__(t):
                    t_left = self.params_by_t[param_key][param].keys()[self.params_by_t[param_key][param].bisect_right(t) - 1]
                    self.params_by_t[param_key][param][t] = self.params_by_t[param_key][param][t_left]
            # multiply
            mults2 = SortedDict({self.date_to_t(d): val for d, val in mults.items()})
            for t in self.params_by_t[param_key][param].keys(): #
                idx_left = mults2.bisect_right(t) - 1
                if idx_left >= 0:  # there is a multiplier that applies to this time.
                    self.params_by_t[param_key][param][t] *= mults2[list(mults2.keys())[idx_left]]

    # set values for a single parameter based on param_tslices
    def set_compartment_param(self, param, attrs: dict=None, vals: dict=None, mults: dict=None, desc=None):
        # get only the compartments we want
        cmpts = ['all'] if attrs is None else self.get_cmpts_matching_attrs(attrs, is_param_cmpts=True)
        # update the parameter
        for cmpt in cmpts:
            self.set_param_by_t(param, cmpt=cmpt, vals=vals, mults=mults)

    # set values for a single parameter based on param_tslices
    def set_from_to_compartment_param(self, param, from_attrs: dict = None, to_attrs: dict = None, vals: dict = None, mults: dict = None, desc=None):
        # get only the compartments we want
        from_cmpts = ['all'] if from_attrs is None else self.get_cmpts_matching_attrs(from_attrs, is_param_cmpts=True)
        # update the parameter
        for from_cmpt in from_cmpts:
            if from_cmpt == 'all':
                to_cmpts = ['all'] if to_attrs is None else self.get_cmpts_matching_attrs(to_attrs, is_param_cmpts=True)
            else:
                to_cmpts = self.update_cmpt_tuple_with_attrs(from_cmpt, to_attrs, is_param_cmpt=True)
            for to_cmpt in to_cmpts:
                self.set_param_by_t(param, from_cmpt=from_cmpt, to_cmpt=to_cmpt, vals=vals, mults=mults)

    # combine param_defs, vaccine_defs, etc. into a time indexed parameters dictionary
    def build_param_lookups(self, apply_vaccines=True, vacc_delay=14):
        logger.debug(f"{str(self.tags)} Building param lookups")
        # clear model params if they exist
        self.params_by_t = {'all': {}}

        for param_def in self.params_defs:
            if 'from_attrs' in param_def:
                self.set_from_to_compartment_param(**param_def)
            else:
                self.set_compartment_param(**param_def)

        # determine all times when params change
        self.params_trange = sorted(list(set.union(*[set(param.keys()) for param_key in self.params_by_t.values() for param in param_key.values()])))
        self.t_prev_lookup = {t_int: max(t for t in self.params_trange if t <= t_int) for t_int in self.trange}

        if apply_vaccines:
            logger.debug(f"{str(self.tags)} Building vaccination param lookups")
            vacc_per_available = self.get_vacc_per_available()

            # apply vacc_delay
            vacc_per_available = vacc_per_available.groupby(['region', 'age']).shift(vacc_delay).fillna(0)

            # group vacc_per_available by trange interval
            #bins = self.params_trange + [self.tend] if self.tend not in self.params_trange else self.params_trange
            bins = list(range(self.tstart, self.tend, 7))
            bins = bins + [self.tend] if self.tend not in bins else bins
            t_index_rounded_down_to_tslices = pd.cut(vacc_per_available.index.get_level_values('t'), bins, right=False, retbins=False, labels=bins[:-1])
            vacc_per_available = vacc_per_available.groupby([t_index_rounded_down_to_tslices, 'region', 'age']).mean()
            vacc_per_available['date'] = [self.t_to_date(d) for d in vacc_per_available.index.get_level_values(0)]
            vacc_per_available = vacc_per_available.reset_index().set_index(['region', 'age']).sort_index()

            # set the fail rate and vacc per unvacc rate for each dose
            for shot in self.attrs['vacc'][1:]:
                for age in self.attrs['age']:
                    for region in self.attrs['region']:
                        vpa_sub = vacc_per_available[['date', shot]].loc[region, age]
                        # TODO: hack for when there's only one date. Is there a better way?
                        if isinstance(vpa_sub, pd.Series):
                            vpa_sub = {vpa_sub[0]: vpa_sub[1]}
                        else:
                            vpa_sub = vpa_sub.reset_index(drop=True).set_index('date').sort_index().drop_duplicates().to_dict()[shot]
                        self.set_compartment_param(param=f'{shot}_per_available', attrs={'age': age, 'region': region}, vals=vpa_sub)

        # Testing changing the vaccination to every week
        self.params_trange = sorted(list(set.union(*[set(param.keys()) for param_key in self.params_by_t.values() for param in param_key.values()])))
        self.t_prev_lookup = {t_int: max(t for t in self.params_trange if t <= t_int) for t_int in self.trange}

    # set TC by slice, and update non-linear multiplier; defaults to reseting the last TC values
    def update_tc(self, tc, replace=True, update_lookup=True):
        if replace:
            self.__tc = tc
        else:
            self.__tc.update(tc)
        if update_lookup:
            self.tc_t_prev_lookup = [max(t for t in self.__tc.keys() if t <= t_int) for t_int in self.trange]  # lookup for latest defined TC value

    def _default_nonlinear_matrix(self):
        return spsp.lil_matrix((self.n_compartments, self.n_compartments))

    # assign default values to matrices
    def reset_ode(self):
        self.terms = []
        self.linear_matrix = {t: spsp.lil_matrix((self.n_compartments, self.n_compartments)) for t in self.params_trange}
        self.nonlinear_matrices = {t: defaultdict(self._default_nonlinear_matrix) for t in self.params_trange}
        self.constant_vector = {t: np.zeros(self.n_compartments) for t in self.params_trange}

    # given a parameter, find its definition in params_by_t for the desired compartment(s). Useful because of the "all" default, so we may have to hunt a bit for the param
    def get_param_key_for_param_and_cmpts(self, param, cmpt=None, from_cmpt=None, to_cmpt=None, nomatch_okay=False, is_param_cmpt=False):
        param_key = None
        if cmpt is not None:
            key = cmpt if is_param_cmpt else tuple(attr for attr, level in zip(cmpt, self.attr_names) if level in self.param_attr_names)
            for key_option in [key, 'all']:
                if key_option in self.params_by_t.keys() and param in self.params_by_t[key_option]:
                    param_key = key_option
                    break
            if param_key is None and not nomatch_okay:
                self.log_and_raise(f"parameter {param} not defined for compartment {cmpt}", ValueError)
        else:
            key0 = from_cmpt if is_param_cmpt else tuple(attr for attr, level in zip(from_cmpt, self.attr_names) if level in self.param_attr_names)
            key1 = to_cmpt if is_param_cmpt else tuple(attr for attr, level in zip(to_cmpt, self.attr_names) if level in self.param_attr_names)
            for key_option in [(key0, key1), ('all', key1), (key0, 'all'), ('all','all')]:
                if key_option in self.params_by_t.keys() and param in self.params_by_t[key_option]:
                    param_key = key_option
                    break
            if param_key is None and not nomatch_okay:
                self.log_and_raise(f"parameter {param} not defined for from-compartment {from_cmpt} and to-compartment {to_cmpt}", ValueError)
        return param_key


    # takes a symbolic expression (coef), and looks up variable names in params to provide a computed output for each t in params_trange
    # cmpt_key may be a single compartment tuple or a tuple of two compartment tuples (for a pair-specific parameter)
    def calc_coef_by_t(self, coef, cmpt=None, from_cmpt=None, to_cmpt=None, lambdified_coef=None):
        if isinstance(coef, dict):
            return {t: coef[t] if t in coef.keys() else 0 for t in self.params_trange}
        elif callable(coef):
            return {t: coef(t) for t in self.params_trange}
        elif coef == '1':
            return {t: 1 for t in self.params_trange}
        elif isinstance(coef, str) or isinstance(coef, sym.Expr):
            coef_by_t = {}
            expr = parse_expr(coef) if isinstance(coef, str) else coef
            relevant_params = [str(s) for s in expr.free_symbols]
            if len(relevant_params) == 1 and coef == relevant_params[0]:
                param_key = self.get_param_key_for_param_and_cmpts(coef, cmpt, from_cmpt, to_cmpt)
                param_vals = {t: self.params_by_t[param_key][coef][t] if self.params_by_t[param_key][coef].__contains__(t) else None for t in self.params_trange}
                for i, t in enumerate(self.params_trange[1:]):
                    if param_vals[t] is None:
                        param_vals[t] = param_vals[self.params_trange[i]]  # carry forward prev value if not defined for this t
                coef_by_t = param_vals
            else:
                func = sym.lambdify(relevant_params, expr) if lambdified_coef is None else lambdified_coef
                param_vals = {t: {} for t in self.params_trange}
                for param in relevant_params:
                    param_key = self.get_param_key_for_param_and_cmpts(param, cmpt, from_cmpt, to_cmpt)
                    for t in self.params_trange:
                        param_vals[t][param] = self.params_by_t[param_key][param][t] if self.params_by_t[param_key][param].__contains__(t) else None
                    for i, t in enumerate(self.params_trange[1:]):
                        if param_vals[t][param] is None:
                            param_vals[t][param] = param_vals[self.params_trange[i]][param]
                for t, tvals in param_vals.items():
                    coef_by_t[t] = func(**tvals)
            return coef_by_t
        else:
            return {t: coef for t in self.params_trange}

    # add a flow term, and add new flow to ODE matrices
    def add_flow_from_cmpt_to_cmpt(self, from_cmpt, to_cmpt, from_coef=None, to_coef=None, from_to_coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None):
        if len(from_cmpt) < len(self.attr_names):
            self.log_and_raise(f'Source compartment `{to_cmpt}` does not have the right number of attributes.', ValueError)
        if len(to_cmpt) < len(self.attr_names):
            self.log_and_raise(f'Destination compartment `{to_cmpt}` does not have the right number of attributes.', ValueError)
        if scale_by_cmpts is not None:
            for cmpt in scale_by_cmpts:
                if len(cmpt) < len(self.attr_names):
                    self.log_and_raise(f'Scaling compartment `{cmpt}` does not have the right number of attributes.', ValueError)

            # retreive the weights for each compartment we are scaling by
            coef_by_t_dl = None
            if scale_by_cmpts_coef:
                # TODO: revisit based on new signature of calc_coef_by_t?
                coef_by_t_lookup = {c: self.calc_coef_by_t(c, to_cmpt) for c in set(scale_by_cmpts_coef)}
                coef_by_t_ld = [coef_by_t_lookup[c] for c in scale_by_cmpts_coef]
                coef_by_t_dl = {t: [dic[t] for dic in coef_by_t_ld] for t in self.params_trange}

        # compute coef by t for the from and to compartments
        coef_by_t = {t: 1 for t in self.params_trange}
        if to_coef is not None:
            parsed_coef = parse_expr(to_coef)  # parsing and lambdifying the coef ahead of time saves doing it for every t
            relevant_params = [str(s) for s in parsed_coef.free_symbols]
            coef_by_t = {t: coef_by_t[t]*coef for t, coef in self.calc_coef_by_t(parse_expr(to_coef), cmpt=to_cmpt, lambdified_coef=sym.lambdify(relevant_params, parsed_coef)).items()}
        if from_coef is not None:
            parsed_coef = parse_expr(from_coef)  # parsing and lambdifying the coef ahead of time saves doing it for every t
            relevant_params = [str(s) for s in parsed_coef.free_symbols]
            coef_by_t = {t: coef_by_t[t]*coef for t, coef in self.calc_coef_by_t(parse_expr(from_coef), cmpt=from_cmpt, lambdified_coef=sym.lambdify(relevant_params, parsed_coef)).items()}
        if from_to_coef is not None:
            parsed_coef = parse_expr(from_to_coef)  # parsing and lambdifying the coef ahead of time saves doing it for every t
            relevant_params = [str(s) for s in parsed_coef.free_symbols]
            coef_by_t = {t: coef_by_t[t]*coef for t, coef in self.calc_coef_by_t(parse_expr(from_to_coef), from_cmpt=from_cmpt, to_cmpt=to_cmpt, lambdified_coef=sym.lambdify(relevant_params, parsed_coef)).items()}

        term = ODEFlowTerm.build(
            from_cmpt_idx=self.cmpt_idx_lookup[from_cmpt],
            to_cmpt_idx=self.cmpt_idx_lookup[to_cmpt],
            coef_by_t=coef_by_t,
            scale_by_cmpts_idxs=[self.cmpt_idx_lookup[cmpt] for cmpt in scale_by_cmpts] if scale_by_cmpts is not None else None,
            scale_by_cmpts_coef_by_t=coef_by_t_dl if scale_by_cmpts is not None else None,
            constant_by_t=self.calc_coef_by_t(constant, to_cmpt) if constant is not None else None)

        # don't even add the term if all its coefficients are zero
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
    def add_flows_from_attrs_to_attrs(self, from_attrs, to_attrs, from_coef=None, to_coef=None, from_to_coef=None, scale_by_attrs=None, scale_by_coef=None, constant=None):
        # Create string summarizing the flow
        from_attrs_str = '(' + ','.join(from_attrs[p] if p in from_attrs.keys() else "*" for p in self.attrs) + ')'
        to_attrs_str = '(' + ','.join(to_attrs[p] if p in to_attrs.keys() else from_attrs[p] if p in from_attrs.keys() else "\"" for p in self.attrs) + ')'
        flow_str = f'{from_attrs_str} -> {to_attrs_str}:'
        if constant is not None:
            flow_str += f' x {constant}'
        if from_coef is not None:
            flow_str += f' x [from comp params: {from_coef}]'
        if to_coef is not None:
            flow_str += f' x [to comp params: {to_coef}]'
        if from_to_coef is not None:
            flow_str += f' x [from/to comp params: {from_to_coef}]'
        if scale_by_attrs:
            scale_by_attrs_str = '(' + ','.join(scale_by_attrs[p] if p in scale_by_attrs.keys() else "*" for p in self.attrs) + ')'
            if scale_by_coef:
                flow_str += f' x [scale by comp: {scale_by_attrs_str}, weighted by {scale_by_coef}]'
            else:
                flow_str += f' x [scale by comp: {scale_by_attrs_str}]'
        self.flows_string += '\n' + flow_str

        scale_by_cmpts = self.get_cmpts_matching_attrs(scale_by_attrs) if scale_by_attrs is not None else None
        from_cmpts = self.get_cmpts_matching_attrs(from_attrs)
        for from_cmpt in from_cmpts:
            to_cmpts = self.update_cmpt_tuple_with_attrs(from_cmpt, to_attrs)
            for to_cmpt in to_cmpts:
                self.add_flow_from_cmpt_to_cmpt(from_cmpt, to_cmpt, from_coef=from_coef, to_coef=to_coef, from_to_coef=from_to_coef, scale_by_cmpts=scale_by_cmpts,
                                                scale_by_cmpts_coef=scale_by_coef, constant=constant)

    # build ODE
    def build_ode_flows(self):
        logger.debug(f"{str(self.tags)} Building ode flows")
        self.flows_string = self.flows_string = '(' + ','.join(self.attr_names) + ')'
        self.reset_ode()

        # vaccination
        logger.debug(f"{str(self.tags)} Building vaccination flows")
        for seir in ['S', 'E', 'A']:
            self.add_flows_from_attrs_to_attrs({'seir': seir, 'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'weak'}, from_coef=f'shot1_per_available * (1 - shot1_fail_rate)')
            self.add_flows_from_attrs_to_attrs({'seir': seir, 'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'none'}, from_coef=f'shot1_per_available * shot1_fail_rate')
            for i in [2, 3]:
                for immun in self.attrs['immun']:
                    # if immun is none, that means that the first vacc shot failed, which means that future shots may fail as well
                    if immun == 'none':
                        self.add_flows_from_attrs_to_attrs({'seir': seir, 'vacc': f'shot{i - 1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, from_coef=f'shot{i}_per_available * (1 - shot{i}_fail_rate / shot{i - 1}_fail_rate)')
                        self.add_flows_from_attrs_to_attrs({'seir': seir, 'vacc': f'shot{i - 1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'none'}, from_coef=f'shot{i}_per_available * (shot{i}_fail_rate / shot{i - 1}_fail_rate)')
                    else:
                        self.add_flows_from_attrs_to_attrs({'seir': seir, 'vacc': f'shot{i - 1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, from_coef=f'shot{i}_per_available')

        # seed variants (only seed the ones in our attrs)
        logger.debug(f"{str(self.tags)} Building seed flows")
        for variant in self.attrs['variant']:
            if variant == 'none':
                continue
            seed_param = f'{variant}_seed'
            from_variant = self.attrs['variant'][0]   # first variant
            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': from_variant, 'immun': 'none'}, {'seir': 'E', 'variant': variant}, constant=seed_param)

        # exposure
        logger.debug(f"{str(self.tags)} Building transmission flows")
        for variant in self.attrs['variant']:
            if variant == 'none':
                continue
            # No mobility between regions (or a single region)
            if self.mobility_mode is None or self.mobility_mode == "none":
                for region in self.attrs['region']:
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef='lamb * betta', from_coef=f'(1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': region})
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef=       'betta', from_coef=f'(1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': region})
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef="lamb * betta", from_coef=f'immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': region})
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef=       "betta", from_coef=f'immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': region})
            # Transmission parameters attached to the susceptible population
            elif self.mobility_mode == "population_attached":
                for infecting_region in self.attrs['region']:
                    for susceptible_region in self.attrs['region']:
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef='lamb * betta', from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       'betta', from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef="lamb * betta", from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       "betta", from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})
            # Transmission parameters attached to the transmission location
            elif self.mobility_mode == "location_attached":
                for infecting_region in self.attrs['region']:
                    for susceptible_region in self.attrs['region']:
                        for transmission_region in self.attrs['region']:
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef='lamb * betta', from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       'betta', from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef="lamb * betta", from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       "betta", from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})

        # disease progression
        logger.debug(f"{str(self.tags)} Building disease progression flows")
        self.add_flows_from_attrs_to_attrs({'seir': 'E'}, {'seir': 'I'}, to_coef='1 / alpha * pS')
        self.add_flows_from_attrs_to_attrs({'seir': 'E'}, {'seir': 'A'}, to_coef='1 / alpha * (1 - pS)')
        # assume no one is receiving both pax and mab
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, to_coef='gamm * hosp * (1 - severe_immunity) * (1 - mab_prev - pax_prev)')
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, to_coef='gamm * hosp * (1 - severe_immunity) * mab_prev * mab_hosp_adj')
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, to_coef='gamm * hosp * (1 - severe_immunity) * pax_prev * pax_hosp_adj')

        # disease termination
        logger.debug(f"{str(self.tags)} Building termination flows")
        for variant in self.attrs['variant']:
            if variant == 'none':
                continue
            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'S', 'immun': 'strong'}, to_coef='gamm * (1 - hosp - dnh) * (1 - priorinf_fail_rate)')
            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'S'}, to_coef='gamm * (1 - hosp - dnh) * priorinf_fail_rate')
            self.add_flows_from_attrs_to_attrs({'seir': 'A', 'variant': variant}, {'seir': 'S', 'immun': 'strong'}, to_coef='gamm * (1 - priorinf_fail_rate)')
            self.add_flows_from_attrs_to_attrs({'seir': 'A', 'variant': variant}, {'seir': 'S'}, to_coef='gamm * priorinf_fail_rate')

            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'immun': 'strong'}, to_coef='1 / hlos * (1 - dh) * (1 - priorinf_fail_rate) * (1-mab_prev)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S'}, to_coef='1 / hlos * (1 - dh) * priorinf_fail_rate * (1-mab_prev)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'immun': 'strong'}, to_coef='1 / (hlos * mab_hlos_adj) * (1 - dh) * (1 - priorinf_fail_rate) * mab_prev')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S'}, to_coef='1 / (hlos * mab_hlos_adj) * (1 - dh) * priorinf_fail_rate * mab_prev')

            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'D'}, to_coef='gamm * dnh * (1 - severe_immunity)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'D'}, to_coef='1 / hlos * dh')

        # immunity decay
        logger.debug(f"{str(self.tags)} Building immunity decay flows")
        for seir in [seir for seir in self.attrs['seir'] if seir != 'D']:
            self.add_flows_from_attrs_to_attrs({'seir': seir, 'immun': 'strong'}, {'immun': 'weak'}, to_coef='1 / imm_decay_days')

    # a matrix which picks out the r
    # possibly could be used in the future to modify TC for subgroups as well e.g. for scenario exploration.
    def build_region_picker_matrix(self):
        logger.debug(f"{str(self.tags)} creating region picker matrix")
        picker = spsp.lil_matrix((self.n_compartments, len(self.regions)))
        for i, region in enumerate(self.regions):
            region_idx = [self.cmpt_idx_lookup[cmpt] for cmpt in self.get_cmpts_matching_attrs({'region': region})]
            picker[region_idx,i] = 1
        self.region_picker_matrix = picker

    # convert ODE matrices to CSR format, to (massively) improve performance
    def compile(self):
        logger.debug(f"{str(self.tags)} compiling ODE")
        for t in self.params_trange:
            self.linear_matrix[t] = self.linear_matrix[t].tocsr()
            for k, v in self.nonlinear_matrices[t].items():
                self.nonlinear_matrices[t][k] = v.tocsr()
        self.region_picker_matrix = self.region_picker_matrix.tocsr()

    # ODE step forward
    def ode(self, t: float, y: list):
        dy = [0] * self.n_compartments
        t_int = self.t_prev_lookup[math.floor(t)]
        t_tc = self.tc_t_prev_lookup[math.floor(t)]
        nlm = [(1 - self.__tc[t_tc][region]) for region in self.regions]
        nlm_vec = self.region_picker_matrix.dot(nlm)

        # apply linear terms
        dy += (self.linear_matrix[t_int]).dot(y)

        # apply non-linear terms
        for scale_by_cmpt_idxs, matrix in self.nonlinear_matrices[t_int].items():
            dy += nlm_vec * sum(itemgetter(*scale_by_cmpt_idxs)(y)) * (matrix).dot(y)

        # apply constant terms
        dy += self.constant_vector[t_int]

        return dy

    # solve ODE using scipy.solve_ivp, and put solution in solution_y
    def solve_seir(self, y0=None, tstart=None, tend=None):
        tstart = self.tstart if tstart is None else tstart
        tend = self.tend if tend is None else tend
        trange = range(tstart, tend + 1) # simulate up to and including tend
        if y0 is None:
            y0 = self.y0_from_dict(self.y0_dict)
        solution = spi.solve_ivp(
            fun=self.ode,
            t_span=[min(trange), max(trange)],
            y0=y0,
            t_eval=trange,
            method=self.ode_method,
            max_step=self.max_step_size
        )
        if not solution.success:
            self.log_and_raise(f'ODE solver failed with message: {solution.message}', RuntimeError)
        # replace the part of the solution we simulated
        self.solution_y[tstart:(tend+1), ] = np.transpose(solution.y)

    # a model must be prepped before it can be run; if any params EXCEPT the TC change, it must be re-prepped
    def prep(self, rebuild_param_lookups=True, **build_param_lookup_args):
        logger.info(f"{str(self.tags)} Prepping Model")
        if rebuild_param_lookups:
            self.build_param_lookups(**build_param_lookup_args)
        self.build_ode_flows()
        self.build_region_picker_matrix()
        self.compile()
        # initialize solution dataframe with all NA values
        self.solution_y = np.ndarray(shape=(len(self.trange), len(self.compartments_as_index)))

    ####################################################################################################################
    ### Reading and Writing Data
    def serialize_vacc(self, df):
        df = df.reset_index()
        df['date'] = [dt.datetime.strftime(d, "%Y-%m-%d") for d in df['date']]
        return df.to_dict('records')

    @classmethod
    def unserialize_vacc(cls, vdict):
        df = pd.DataFrame.from_dict(vdict)
        df['date'] = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in df['date']]
        return df.set_index(['date', 'region', 'age'])

    def serialize_hosp(self, df):
        df = df.reset_index()
        df['date'] = [dt.datetime.strftime(d, "%Y-%m-%d") for d in df['date']]
        return df.to_dict('records')

    @classmethod
    def unserialize_hosp(cls, hdict):
        df = pd.DataFrame.from_dict(hdict)
        df['date'] = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in df['date']]
        return df.set_index(['region', 'date']).sort_index()

    def serialize_mob(self, df):
        df = df.reset_index()
        df['date'] = [dt.datetime.strftime(d, "%Y-%m-%d") for d in df['date']]
        return df.to_dict('records')

    @classmethod
    def unserialize_mob(cls, mdict):
        df = pd.DataFrame.from_dict(mdict)
        df['date'] = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in df['date']]
        return df.set_index(['date', 'from_region', 'to_region']).sort_index()

    @classmethod
    def unserialize_tc(cls, tc_dict):
        # JSON can't handle numbers as dict keys
        return {int(key): val for key, val in tc_dict.items()}

    def serialize_y0_dict(self, y0_dict):
        return [list(key) + [val] for key, val in y0_dict.items()]

    @classmethod
    def unserialize_y0_dict(cls, y0_list):
        return {tuple(li[:-1]): float(li[-1]) for li in y0_list} if y0_list is not None else None

    # serializes SOME of this model's properties to a json format which can be written to database
    # model needs prepping still
    def to_json_string(self):
        logger.debug(f"{str(self.tags)} Serializing model to json")
        keys = ['base_spec_id', 'spec_id', 'region_fit_spec_ids', 'region_fit_result_ids', 'tags', '_CovidModel__start_date', '_CovidModel__end_date', '__attrs', '_CovidModel__tc', 'tc_cov', 'tc_t_prev_lookup', '_CovidModel__params_defs',
                '_CovidModel__region_defs', '_CovidModel__regions', '_CovidModel__vacc_proj_params', '_CovidModel__mobility_mode', 'actual_mobility', 'proj_mobility', 'proj_mobility', 'mobility_proj_params', 'actual_vacc_df', 'proj_vacc_df',  'hosps', '_CovidModel__hosp_reporting_frac',
                '_CovidModel__y0_dict', 'max_step_size']
        # add in proj_mobility
        serial_dict = OrderedDict()
        for key in keys:
            val = self.__dict__[key]
            if isinstance(val, dt.date):
                serial_dict[key] = val.strftime('%Y-%m-%d')
            elif isinstance(val, np.ndarray):
                serial_dict[key] = val.tolist()
            elif key in ['actual_vacc_df', 'proj_vacc_df'] and val is not None:
                serial_dict[key] = self.serialize_vacc(val)
            elif key in ['actual_mobility', 'proj_mobility'] and val is not None:
                serial_dict[key] = self.serialize_mob(val)
            elif key == 'hosps' and val is not None:
                serial_dict[key] = self.serialize_hosp(val)
            elif key == '_CovidModel__y0_dict' and self.__y0_dict is not None:
                serial_dict[key] = self.serialize_y0_dict(val)
            elif key == 'max_step_size':
                serial_dict[key] = val if not np.isinf(val) else 'inf'
            else:
                serial_dict[key] = val
        return json.dumps(serial_dict)

    def from_json_string(self, s):
        logger.debug(f"{str(self.tags)} repopulating model from serialized json")
        raw = json.loads(s)
        for key, val in raw.items():
            if key in ['_CovidModel__start_date', '_CovidModel__end_date']:
                self.__dict__[key] = dt.datetime.strptime(val, "%Y-%m-%d").date()
            elif key == '_CovidModel__tc':
                self.__dict__[key] = CovidModel.unserialize_tc(val)
            elif key == "tc_cov":
                #TODO
                pass
            elif key in ['actual_vacc_df', 'proj_vacc_df'] and val is not None:
                self.__dict__[key] = CovidModel.unserialize_vacc(val)
            elif key in ['actual_mobility', 'proj_mobility'] and val is not None:
                self.__dict__[key] = CovidModel.unserialize_mob(val)
            elif key == 'hosp' and val is not None:
                self.__dict__[key] = CovidModel.unserialize_hosp(val)
            elif key == '_CovidModel__y0_dict':
                self.__dict__[key] = self.unserialize_y0_dict(val)
            elif key == 'max_step_size':
                self.__dict__[key] = np.inf if val == 'inf' else float(val)
            else:
                self.__dict__[key] = val

        # triggers updating of tend, trange, etc.
        self.end_date = self.end_date

    def write_specs_to_db(self, engine):
        if engine is None:
            engine = db_engine()
        logger.debug(f"{str(self.tags)} writing specs to db")
        # returns all the data you would need to write to the database but doesn't actually write to the database

        with Session(engine) as session:
            # generate a spec_id so we can assign it to ourselves
            specs_table = get_sqa_table(engine, schema='covid_model', table='specifications')
            max_spec_id = session.query(func.max(specs_table.c.spec_id)).scalar()
            self.spec_id = max_spec_id + 1

            stmt = specs_table.insert().values(OrderedDict([
                ("base_spec_id", int(self.base_spec_id) if self.base_spec_id is not None else None),
                ("spec_id", self.spec_id),
                ("created_at", dt.datetime.now()),
                ("start_date", self.start_date),
                ("end_date", self.end_date),
                ("tags", json.dumps(self.tags)),
                ("regions", json.dumps(self.regions)),
                ("tslices", list(self.__tc.keys())),
                #("tc", list(self.__tc.values())), # doesn't work now that TC is a nested dict, This field in the DB is a list.
                ("serialized_model", self.to_json_string())
            ]))
            session.execute(stmt)
            session.commit()
        logger.debug(f"{str(self.tags)} spec_id: {self.spec_id}")

    def read_from_base_spec_id(self, engine):
        if engine is None:
            engine = db_engine()
        df = pd.read_sql_query(f"select * from covid_model.specifications where spec_id = {self.base_spec_id}", con=engine, coerce_float=True)
        if len(df) == 0:
            self.log_and_raise(f'{self.base_spec_id} is not a valid spec ID.', ValueError)
        row = df.iloc[0]
        self.from_json_string(row['serialized_model'])
        # The spec ID we pulled from the DB becomes the base_spec_id (will match what self.base_spec_id was before calling this function)
        self.base_spec_id = self.spec_id
        self.spec_id = None

    def _col_to_json(self, d):
        return json.dumps(d, ensure_ascii=False)

    def write_results_to_db(self, engine, new_spec=False, vals_json_attr='seir', cmpts_json_attrs=('region', 'age', 'vacc')):
        if engine is None:
            engine = db_engine()
        logger.debug(f"{str(self.tags)} writing results to db")
        table = 'results_v2'
        # if there's no existing spec_id assigned, write specs to db to get one
        if self.spec_id is None or new_spec:
            self.write_specs_to_db(engine)

        # build data frame with index of (t, region, age, vacc) and one column per seir cmpt
        solution_sum_df = self.solution_sum_df([vals_json_attr] + list(cmpts_json_attrs)).stack(cmpts_json_attrs)

        # build export dataframe
        df = pd.DataFrame(index=solution_sum_df.index)
        df['date'] = solution_sum_df.index.get_level_values('date')
        df['cmpt'] = solution_sum_df.index.droplevel('date').to_frame().to_dict(
            orient='records') if solution_sum_df.index.nlevels > 1 else None
        df['vals'] = solution_sum_df.to_dict(orient='records')
        for col in ['cmpt', 'vals']:
            df[col] = df[col].map(self._col_to_json)

        # if a sim_id is provided, insert it as a simulation result; some fields are different
        # build unique parameters dataframe
        df['created_at'] = dt.datetime.now()
        df['spec_id'] = self.spec_id
        df['result_id'] = pd.read_sql(f'select coalesce(max(result_id), 0) from covid_model.{table}', con=engine).values[0][0] + 1

        # write to database
        chunksize = int(np.floor(5000.0 / df.shape[1]))  # max parameters is 10,000. Assume 1 param per column and give some wiggle room because 10,000 doesn't always work

        results = df.to_sql(table, con=engine, schema='covid_model', index=False, if_exists='append', method='multi', chunksize=chunksize)

        self.result_id = df['result_id'][0]
        return df
