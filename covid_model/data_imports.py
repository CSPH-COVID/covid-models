import datetime as dt
import json

import numpy as np
import pandas as pd

from covid_model.utils import get_params


def normalize_date(date):
    return date if type(date) == dt.date or date is None else date.date()


class ExternalData:
    def __init__(self, engine=None, t0_date=None, fill_from_date=None, fill_to_date=None):
        self.engine = engine
        self.t0_date = normalize_date(t0_date) if t0_date is not None else None
        self.fill_from_date = normalize_date(fill_from_date) if fill_from_date is not None else normalize_date(t0_date)
        self.fill_to_date = normalize_date(fill_to_date) if fill_to_date is not None else None

    def fetch(self, fpath=None, rerun=True, **args):
        if rerun:
            df = self.fetch_from_db(**args)
            if fpath is not None:
                df.reset_index().drop(columns='index', errors='ignore').to_csv(fpath, index=False)
        else:
            df = pd.read_csv(fpath)

        if self.t0_date is not None:
            index_names = [idx for idx in df.index.names if idx not in (None, 'measure_date')]
            df = df.reset_index()
            df['t'] = (pd.to_datetime(df['measure_date']).dt.date - self.t0_date).dt.days
            min_t = min(df['t'])
            max_t = max(df['t'])
            df = df.reset_index().drop(columns=['index', 'level_0', 'measure_date'], errors='ignore').set_index(['t'] + index_names)

            trange = range((self.fill_from_date - self.t0_date).days, (self.fill_to_date - self.t0_date).days + 1 if self.fill_to_date is not None else max_t)
            index = pd.MultiIndex.from_product([trange] + [df.index.unique(level=idx) for idx in index_names]).set_names(['t'] + index_names) if index_names else range(max_t)
            empty_df = pd.DataFrame(index=index)
            df = empty_df.join(df, how='left').fillna(0)

        return df

    def fetch_from_db(self, **args) -> pd.DataFrame:
        # return pd.read_sql(args['sql'], self.engine)
        return pd.read_sql(con=self.engine, **args)


class ExternalHosps(ExternalData):
    def fetch_from_db(self):
        return pd.read_sql('select * from cdphe.emresource_hospitalizations', self.engine, parse_dates=['measure_date'])


class ExternalVacc(ExternalData):
    def fetch_from_db(self, county_ids=None):
        if county_ids is None:
            sql = open('sql/vaccination_by_age_group_with_boosters_wide.sql', 'r').read()
            return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'])
        else:
            county_ids = county_ids if type(county_ids) == list else [county_ids]
            sql = open('sql/vaccination_by_age_group_with_boosters_wide_county_subset.sql', 'r').read()
            return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'], params={'county_ids': county_ids})


class ExternalVaccWithProjections(ExternalData):
    def fetch_from_db(self, proj_params=None, group_pop=None):
        sql = open('sql/vaccination_by_age_group_with_boosters_wide.sql', 'r').read()

        proj_params = proj_params if type(proj_params) == dict else json.load(open(proj_params))
        proj_lookback = proj_params['lookback'] if 'lookback' in proj_params.keys() else 7
        proj_fixed_rates = proj_params['fixed_rates'] if 'fixed_rates' in proj_params.keys() else None
        max_cumu = proj_params['max_cumu'] if 'max_cumu' in proj_params.keys() else 0
        max_rate_per_remaining = proj_params['max_rate_per_remaining'] if 'max_rate_per_remaining' in proj_params.keys() else 1.0
        realloc_priority = proj_params['realloc_priority'] if 'realloc_priority' in proj_params.keys() else None

        df = pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'])
        shots = list(df.columns)

        # add projections
        proj_from_date = df.index.get_level_values('measure_date').max() + dt.timedelta(days=1)
        if self.fill_to_date >= proj_from_date:
            proj_date_range = pd.date_range(proj_from_date, self.fill_to_date)
            # project rates based on the last {proj_lookback} days of data
            projected_rates = df.loc[(proj_from_date - dt.timedelta(days=proj_lookback)):].groupby('age').sum() / float(proj_lookback)
            # override rates using fixed values from proj_fixed_rates, when present
            if proj_fixed_rates:
                for shot in shots:
                    projected_rates[shot] = pd.DataFrame(proj_fixed_rates)[shot]
            # build projections
            projections = pd.concat({d.date(): projected_rates for d in proj_date_range}).rename_axis(index=['measure_date', 'age'])

            # reduce rates to prevent cumulative vaccination from exceeding max_cumu
            if max_cumu:
                cumu_vacc = df.groupby('age').sum()
                groups = realloc_priority if realloc_priority else projections.index.unique('age')
                # vaccs = df.index.unique('vacc')
                for d in projections.index.unique('measure_date'):
                    this_max_cumu = get_params(max_cumu.copy(), (d - self.fill_from_date).days)
                    max_cumu_df = pd.DataFrame(this_max_cumu) * pd.DataFrame(group_pop, index=shots).transpose()
                    for i in range(len(groups)):
                        group = groups[i]
                        current_rate = projections.loc[(d, group)]
                        max_rate = max_rate_per_remaining * (max_cumu_df.loc[group] - cumu_vacc.loc[group])
                        excess_rate = (projections.loc[(d, group)] - max_rate).clip(lower=0)
                        projections.loc[(d, group)] -= excess_rate
                        # if a reallocate_order is provided, reallocate excess rate to other groups
                        if i < len(groups) - 1 and realloc_priority is not None:
                            projections.loc[(d, groups[i + 1])] += excess_rate

                    cumu_vacc += projections.loc[d]

            df = pd.concat({False: df, True: projections}).rename_axis(index=['is_projected', 'measure_date', 'age']).reorder_levels(['measure_date', 'age', 'is_projected']).sort_index()

        return df


class ExternalContactMatrices(ExternalData):
    pass


# load actual hospitalization data for fitting
# def get_hosps(engine, min_date=dt.datetime(2020, 1, 24)):
#     actual_hosp_df = pd.read_sql(open('sql/emresource_hospitalizations.sql').read(), engine)
#     actual_hosp_df['t'] = ((pd.to_datetime(actual_hosp_df['measure_date']) - min_date) / np.timedelta64(1, 'D')).astype(int)
#     actual_hosp_tmin = actual_hosp_df[actual_hosp_df['currently_hospitalized'].notnull()]['t'].min()
#     return [0] * actual_hosp_tmin + list(actual_hosp_df['currently_hospitalized'])


def get_hosps_df(engine):
    return pd.read_sql(open('sql/emresource_hospitalizations.sql').read(), engine, parse_dates=['measure_date']).set_index('measure_date')['currently_hospitalized']


def get_hosps_by_age(engine, fname):
    df = pd.read_csv(fname, parse_dates=['dates']).set_index('dates')
    df = df[[col for col in df.columns if col[:17] == 'HospCOVIDPatients']]
    df = df.rename(columns={col: col.replace('HospCOVIDPatients', '').replace('to', '-').replace('plus', '+') for col in df.columns})
    df = df.stack()
    df.index = df.index.set_names(['measure_date', 'age'])
    cophs_total = df.groupby('measure_date').sum()
    emr_total = get_hosps_df(engine)
    return df * emr_total / cophs_total


# load actual death data for plotting
def get_deaths(engine, min_date=dt.datetime(2020, 1, 24)):
    sql = """
        select 
            reporting_date as measure_date
            , sum(total_count) as new_deaths
        from cdphe.covid19_county_summary ccs 
        where count_type = 'deaths'
        group by 1
        order by 1"""
    df = pd.read_sql(sql, engine, parse_dates=['measure_date']).set_index('measure_date')
    df = pd.date_range(min_date, df.index.max()).to_frame().join(df, how='left').drop(columns=[0]).fillna(0)
    df['cumu_deaths'] = df['new_deaths'].cumsum()

    return df


def get_deaths_by_age(engine):
    sql = """select
        date::date as measure_date
        , case when age_group in ('0-5', '6-11', '12-17', '18-19') then '0-19' else age_group end as "group"
        , sum(count::int) as new_deaths
    from cdphe.temp_covid19_county_summary
    where count_type like 'deaths, %%' and date_type = 'date of death'
    group by 1, 2
    order by 1, 2"""
    df = pd.read_sql(sql, engine, parse_dates=['measure_date']).set_index(['measure_date', 'age'])
    return df


def get_vaccinations_by_county(engine):
    sql = open('sql/vaccinations_by_age_by_county.sql', 'r').read()
    df = pd.read_sql(sql, engine)
    return df


def get_corrected_emresource(fpath):
    raw_hosps = pd.read_excel(fpath, 'COVID hospitalized_confirmed', engine='openpyxl', index_col='Resource facility name').drop(index='Grand Total').rename(columns=pd.to_datetime).stack()
    raw_hosps.index = raw_hosps.index.set_names(['facility', 'date'])

    raw_reports = pd.read_excel(fpath, 'Latest EMR update', engine='openpyxl', index_col='Resource facility name').drop(index='Grand Total').rename(columns=pd.to_datetime).stack()
    raw_reports.index = raw_reports.index.set_names(['facility', 'date'])
    raw_reports = pd.to_datetime(raw_reports).rename('last_report_date').sort_index()

    print(raw_reports)
    print(pd.to_datetime(pd.to_numeric(raw_reports).groupby('facility').rolling(20).agg(np.max)))
