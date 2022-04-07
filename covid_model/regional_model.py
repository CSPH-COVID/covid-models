import numpy as np
from scipy import sparse

from collections import OrderedDict
from covid_model.model import CovidModel
from covid_model.data_imports import get_region_mobility_from_file, get_region_mobility_from_db
from covid_model.db import db_engine


class RegionalCovidModel(CovidModel):
    attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                        'priorinf': ['none', 'non-omicron', 'omicron'],
                        'variant': ['none', 'alpha', 'delta', 'omicron'],
                        'immun': ['none', 'weak', 'strong'],
                        'region': ['']})

    param_attr_names = ('age', 'vacc', 'priorinf', 'variant', 'immun', 'region')

    def __init__(self, region=None, *args, **kwargs):
        self.attr['region'] = [region] if region is not None else None
        super(RegionalCovidModel, self).__init__(*args, **kwargs)

    @property
    def y0_dict(self):
        y0d = {('S', age, 'none', 'none', 'none', 'none', self.attr['region'][0]): n for age, n in self.group_pops.items()}
        return y0d

    @classmethod
    def construct_region_contact_matrices(cls, regions: OrderedDict, region_params, fpath=None):
        regions = OrderedDict([(key, (regions[key], val['county_fips'], val['county_names'], val['total_pop'])) for key, val in region_params.items() if key in regions.keys()])
        df = get_region_mobility_from_file(fpath) if fpath else get_region_mobility_from_db(db_engine())

        # add regions to dataframe
        regions_lookup = {val: key for key, vals in regions.items() for val in vals[1]}
        df['origin_region'] =      [regions_lookup[id] if id is not None and id in regions_lookup.keys() else None for id in df['origin_county_id']]
        df['destination_region'] = [regions_lookup[id] if id is not None and id in regions_lookup.keys() else None for id in df['destination_county_id']]
        df = df.dropna()

        df = df.drop(['origin_county_id', 'destination_county_id'], axis=1)\
            .groupby(['measure_date', 'origin_region', 'destination_region'])\
            .aggregate(total_dwell_duration_hrs=('total_dwell_duration_hrs', 'sum'))

        # Create dictionaries of matrices, both D and M.
        dates = df.index.get_level_values('measure_date')
        region_idx = {region: i for i, region in enumerate(regions.keys())}
        dwell_matrices = {}

        for date in dates:
            dfsub = df.loc[df.index.get_level_values('measure_date') == date].reset_index('measure_date', drop=True).reset_index()
            idx_i = [region_idx[region] for region in dfsub['origin_region']]
            idx_j = [region_idx[region] for region in dfsub['destination_region']]
            vals  = dfsub['total_dwell_duration_hrs']
            dwell = sparse.coo_array((vals, (idx_i, idx_j)), shape=(len(regions), len(regions))).todense()
            dwell[np.isnan(dwell)] = 0
            dwell_rownorm = dwell / dwell.sum(axis=1)[:, np.newaxis]
            dwell_colnorm = dwell / dwell.sum(axis=0)[np.newaxis, :]
            dwell_matrices[date] = {"dwell": dwell, "dwell_rownorm": dwell_rownorm, "dwell_colnorm": dwell_colnorm}

        return {"regions": regions, "df": df, "dwell_matrices": dwell_matrices}
