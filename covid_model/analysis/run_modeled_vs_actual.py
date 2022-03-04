import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import datetime as dt
import json
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates, format_date_axis
from time import perf_counter
from timeit import timeit
import argparse

from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser
# from covid_model.model_with_omicron import CovidModelWithVariants
# from covid_model.model_with_immunity_rework import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications
from covid_model.run_model_scenarios import build_legacy_output_df

if __name__ == '__main__':
    parser = ModelSpecsArgumentParser()

    print('Prepping model...')
    t0 = perf_counter()
    engine = db_engine()
    model = CovidModel(engine=engine, **parser.specs_args_as_dict())
    model.prep()
    t1 = perf_counter()
    print(f'Model prepped in {t1-t0} seconds.')

    print('Running model...')
    t0 = perf_counter()
    model.solve_seir()
    t1 = perf_counter()
    print(f'Model run in {t1-t0} seconds.')

    fig, ax = plt.subplots()
    actual_hosps(engine, county_ids=model.tags['county_ids'] if 'county_ids' in model.tags.keys() else None, ax=ax)
    modeled(model, 'Ih', ax=ax)

    plt.show()