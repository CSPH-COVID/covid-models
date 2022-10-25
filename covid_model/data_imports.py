""" Python Standard Library """
import datetime as dt
import json
""" Third Party Imports """
import numpy as np
import pandas as pd
""" Local Imports """


def normalize_date(date):
    """Convert datetime to date if necessary

    Args:
        date: either a dt.datetime.date or dt.datetime object

    Returns: dt.datetime.date object

    """
    return date if type(date) == dt.date or date is None else date.date()


class ExternalData:
    """Base class for loading external data, either from file or database

    """
    def __init__(self, engine=None, t0_date=None, fill_from_date=None, fill_to_date=None):
        self.engine = engine
        self.t0_date = normalize_date(t0_date) if t0_date is not None else None
        self.fill_from_date = normalize_date(fill_from_date) if fill_from_date is not None else normalize_date(t0_date)
        self.fill_to_date = normalize_date(fill_to_date) if fill_to_date is not None else None

    def fetch(self, fpath=None, rerun=True, **args):
        """Template function for retrieving data and optionally saving to a file

        Args:
            fpath: optional location to save data
            rerun: whether to fetch the data again
            **args: additional arguments passed to self.fetch_from_db

        Returns:

        """
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
        """fetch data from database and return Pandas Dataframe

        Args:
            **args: arguments for pandas.read_sql function

        Returns: pandas dataframe of loaded data

        """
        # return pd.read_sql(args['sql'], self.engine)
        return pd.read_sql(con=self.engine, **args)


class ExternalHospsEMR(ExternalData):
    """Class for Retrieving EMResource hospitalization data from database

    """
    def fetch_from_db(self):
        """Retrieve hospitalization data from database using query in emresource_hospitalizations.sql

        Returns: Pandas dataframe of hospitalization data

        """
        sql = open('covid_model/sql/emresource_hospitalizations.sql', 'r').read()
        return pd.read_sql(sql, self.engine, index_col=['measure_date'])


class ExternalHospsCOPHS(ExternalData):
    """Class for Retrieving COPHS hospitalization data from database

    """
    def fetch_from_db(self, region_ids: list):
        """Retrieve hospitalization data from database using query in hospitalized_county_subset.sql

        COPHS contains county level hospitalizations, so optionally you can specify a subset of counties to query for.

        Args:
            county_ids: list of county FIPS codes that you want hospitalization data for

        Returns: Pandas dataframe of hospitalization data

        """
        sql = open('covid_model/sql/hospitalized_region_subset.sql', 'r').read()
        return pd.read_sql(sql, self.engine, index_col=['measure_date'], params={'region_ids': region_ids})


class ExternalVacc(ExternalData):
    """Class for retrieving vaccinations data from database

    """
    def fetch_from_db(self, county_ids: list=None):
        """Retrieve vaccinations from database using query in sql file, either for entire state or for a subset of counties

        Args:
            county_ids: list of county FIPS codes that you want vaccinations for (optional)

        Returns:

        """
        if county_ids is None:
            sql = open('covid_model/sql/vaccination_by_age_group_with_boosters_wide.sql', 'r').read()
            return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'])
        else:
            sql = open('covid_model/sql/vaccination_by_age_group_with_boosters_wide.sql', 'r').read()
            return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'])
            #sql = open('covid_model/sql/vaccination_by_age_group_with_boosters_wide_county_subset.sql', 'r').read()
            #return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'], params={'county_ids': county_ids})

def get_region_mobility_from_db(engine, county_ids=None, fpath=None) -> pd.DataFrame:
    """Standalone function to retrieve mobility data from database and possibly write to a file

    Args:
        engine: connection to database
        county_ids: list of FIPS codes to retrieve mobility data for
        fpath: file path to save the mobility data once downloaded (optional)

    Returns:

    """
    if county_ids is None:
        with open('covid_model/sql/mobility_dwell_hours.sql') as f:
            df = pd.read_sql(f.read(), engine, index_col=['measure_date'])
    else:
        with open('covid_model/sql/mobility_dwell_hours_county_subset.sql') as f:
            df = pd.read_sql(f.read(), engine, index_col=['measure_date'], params={'county_ids': county_ids})
    if fpath:
        df.to_csv(fpath)
    return df
