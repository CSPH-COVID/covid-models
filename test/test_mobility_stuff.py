import os
import json
import unittest
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from data_imports import get_region_mobility_from_db, get_region_mobility_from_file
from db import db_engine
from covid_model.regional_model import RegionalCovidModel


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # load credentials and set environment variables
        with open('../creds.json') as creds_file:
            for cred_key, cred_val in json.load(creds_file).items():
                os.environ[cred_key] = cred_val

    def test_load_region_mobility_from_db(self):
        os.chdir('../covid_model')
        engine = db_engine()
        df = get_region_mobility_from_db(engine, '../contact_matrices/mobility.csv')

    def test_load_region_mobility_from_file(self):
        os.chdir('../covid_model')
        df = get_region_mobility_from_file('../contact_matrices/mobility.csv')

    def test_create_region_contact_matrix(self):
        os.chdir('../covid_model')
        # list of regions to run in the regional model
        regions = OrderedDict([
            ("cent", "Central"),
            ("cm", "Central Mountains"),
            ("met", "Metro"),
            ("ms", "Metro South"),
            ("ne", "Northeast"),
            ("nw", "Northwest"),
            ("slv", "San Luis Valley"),
            ("sc", "South Central"),
            ("sec", "Southeast Central"),
            ("sw", "Southwest"),
            ("wcp", "West Central Partnership")
        ])
        # get region-to-county mapping by loading the region_params.json file
        with open('input/region_params.json', 'r') as f:
            region_params = json.loads(f.read())
        regions = OrderedDict([(key, (regions[key], val['county_fips'], val['county_names'], val['total_pop'])) for key, val in region_params.items() if key in regions.keys()])
        cm = RegionalCovidModel.construct_region_contact_matrices(regions, fpath='../contact_matrices/mobility.csv')
        dates = list(cm['dwell_matrices'].keys())

        # what's the max contact matrix value?
        max_dwell = max([M['dwell'].max() for M in cm['dwell_matrices'].values()])

        # make some heatmaps

        region_names = [val[0] for val in regions.values()]
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            D = cm['dwell_matrices'][date]['dwell']
            Dr= cm['dwell_matrices'][date]['dwell_rownorm']
            Dc= cm['dwell_matrices'][date]['dwell_colnorm']
            M = np.dot(Dr, np.transpose(Dc))
            for mat, name in zip([D, Dr, Dc, M], ["Dwell Time", "Row Normalized Dwell Time", "Column Normalized Dwell Time", "Contact Matrix"]):
                fig_dim = 12 if name == "Dwell Time" else 9
                fmt = ".2g" if name == "Dwell Time" else ".2f"
                vmax = max_dwell if name == "Dwell Time" else 1
                fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=300)
                sns.heatmap(mat, vmin=0, vmax=vmax, annot=True, fmt=fmt, square=True, xticklabels=region_names, yticklabels=region_names)
                ax.figure.tight_layout()
                ax.set_title(f'{name} For Week Of {date_str}')
                plt.savefig(f'output/{name.replace(" ", "_").lower()}_{date_str}')
                plt.close()


if __name__ == '__main__':
    unittest.main()