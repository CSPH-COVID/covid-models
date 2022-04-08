### Python Standard Library ###
import json
from collections import OrderedDict
### Third Party Imports ###
### Local Imports ###
from covid_model.db import db_engine
from covid_model.data_imports import ExternalHosps

regions = OrderedDict([("cent", "Central"),
            ("cm", "Central Mountains"),
            ("met", "Metro"),
            ("ms", "Metro South"),
            ("ne", "Northeast"),
            ("nw", "Northwest"),
            ("slv", "San Luis Valley"),
            ("sc", "South Central"),
            ("sec", "Southeast Central"),
            ("sw", "Southwest"),
            ("wcp", "West Central Partnership")])


def get():
    for region in regions.keys():
        region_county_ids = json.load(open('input/region_params.json'))[region]['county_fips']
        hosps = ExternalHosps(db_engine()).fetch_from_db(region_county_ids)
        start_date = min(hosps.index[hosps['currently_hospitalized'] > 0]).strftime("%Y-%m-%d")
        print(region + '    "initial_seed": {"tslices": ["' + start_date + '"], "value": [2.2, 0]},')


if __name__ == "__main__":
    get()
