### Python Standard Library ###
import json
import datetime as dt
import argparse
### Third Party Imports ###
import numpy as np
### Local Imports ###
from covid_model.db import db_engine
from covid_model.cli_specs import ModelSpecsArgumentParser
from covid_model.model_specs import CovidModelSpecifications
from covid_model.model_sims import CovidModelSimulation


if __name__ == '__main__':
    parser = ModelSpecsArgumentParser()
    parser.add_argument("-n", "--number_of_sims", type=int, help="number of simulations to run")
    run_params = parser.parse_args()

    engine = db_engine()

    specs = CovidModelSpecifications(engine=engine, **parser.specs_args_as_dict())
    specs.spec_id = None
    specs.tags['run_type'] = 'sim'

    specs.write_to_db(engine)
    sims = CovidModelSimulation(specs, engine=engine, end_date=specs.end_date)
    sims.run_base_result()
    sims.run_simulations(run_params.number_of_sims, sims_per_fitted_sample=10)
