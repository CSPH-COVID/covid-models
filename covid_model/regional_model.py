import numpy as np
from scipy import sparse

from collections import OrderedDict
from covid_model.model import CovidModel
from covid_model.data_imports import get_region_mobility_from_file, get_region_mobility_from_db
from db import db_engine


class RegionalCovidModel(CovidModel):
    attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                        'priorinf': ['none', 'non-omicron', 'omicron'],
                        'variant': ['none', 'alpha', 'delta', 'omicron'],
                        'immun': ['none', 'weak', 'strong'],
                        'region': None})

    param_attr_names = ('age', 'vacc', 'priorinf', 'variant', 'immun')

    @classmethod
    def construct_region_contact_matrix(cls, regions: OrderedDict, fpath=None):
        df = get_region_mobility_from_file(fpath) if fpath else get_region_mobility_from_db(db_engine())

        # add regions to dataframe
        regions_lookup = {val: key for key, vals in regions.items() for val in vals[1]}
        df['origin_region'] =      [regions_lookup[id] if id is not None and id in regions_lookup.keys() else None for id in df['origin_county_id']]
        df['destination_region'] = [regions_lookup[id] if id is not None and id in regions_lookup.keys() else None for id in df['destination_county_id']]
        df = df.dropna()

        df = df.drop(['origin_county_id', 'destination_county_id'], axis=1)\
            .groupby(['measure_date', 'origin_region', 'destination_region'])\
            .aggregate(total_dwell_duration_hrs=('total_dwell_duration_hrs', 'sum'))

        # compute dwell share between regions
        df_origin = df.groupby(['measure_date', 'origin_region']).aggregate(origin_grand_total_dwell_duration_hrs=('total_dwell_duration_hrs', 'sum'))
        df = df.join(df_origin, on=['measure_date', 'origin_region'])
        df['share_total_dwell_hrs'] = df['total_dwell_duration_hrs'] / df['origin_grand_total_dwell_duration_hrs']
        df = df.drop(['total_dwell_duration_hrs', 'origin_grand_total_dwell_duration_hrs'], axis=1)

        # Create dictionaries of matrices, both D and M.
        dates = df.index.get_level_values('measure_date')
        region_idx = {region: i for i, region in enumerate(regions.keys())}
        Ds = {}
        for date in dates:
            dfsub = df.loc[df.index.get_level_values('measure_date') == date].reset_index('measure_date', drop=True).reset_index()
            idx_i = [region_idx[region] for region in dfsub['origin_region']]
            idx_j = [region_idx[region] for region in dfsub['destination_region']]
            vals  = dfsub['share_total_dwell_hrs']
            Ds[date] = sparse.coo_array((vals, (idx_i, idx_j)), shape=(len(regions), len(regions))).todense()
            Ds[date][np.isnan(Ds[date])] = 0
        Ms = {}
        for date, D in Ds.items():
            Ms[date] = np.dot(D, np.transpose(D))

        return {"regions": regions, "df": df, "D": Ds, "M": Ms}



