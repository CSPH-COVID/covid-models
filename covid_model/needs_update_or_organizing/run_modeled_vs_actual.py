### Python Standard Library ###
from time import perf_counter
### Third Party Imports ###
import matplotlib.pyplot as plt
### Local Imports ###
from covid_model.analysis.charts import modeled, actual_hosps
from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser


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
