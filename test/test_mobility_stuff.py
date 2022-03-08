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
        cm = RegionalCovidModel.construct_region_contact_matrix(regions, fpath='../contact_matrices/mobility.csv')
        dates = list(cm['D'].keys())

        # what's the max contact matrix value?
        max_contact = max([M.max() for M in cm['M'].values()])


        # make some heatmaps
        h = 8
        w = 8
        region_names = [val[0] for val in regions.values()]
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            fig, ax = plt.subplots(figsize=(w, h), dpi=300)
            sns.heatmap(cm['D'][date], vmin=0, vmax=1, annot=True, fmt=".2f", square=True, xticklabels=region_names, yticklabels=region_names)
            ax.set_title(f'Mobility For Week Of {date_str}')
            plt.savefig(f'output/mobility_{date_str}')

            fig, ax = plt.subplots(figsize=(w, h), dpi=300)
            sns.heatmap(cm['M'][date], vmin=0, vmax=max_contact, annot=True, fmt=".2f", square=True, xticklabels=region_names, yticklabels=region_names)
            ax.set_title(f'Contact Matrix For Week Of {date_str}')
            plt.savefig(f'output/contact_matrix_{date_str}')



if __name__ == '__main__':
    unittest.main()