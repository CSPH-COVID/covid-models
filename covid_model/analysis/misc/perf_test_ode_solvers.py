import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import json
from covid_model.analysis.charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates
from time import perf_counter
from timeit import timeit

from covid_model.db import db_engine
from covid_model.model import CovidModel
# from covid_model.model_with_omicron import CovidModelWithVariants
from covid_model.model_with_immunity_rework import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications

if __name__ == '__main__':
    engine = db_engine()

    # model = CovidModelWithVariants()
    model = CovidModel(end_date=dt.date(2022, 1, 1))

    print('Prepping model...')
    print(timeit("model.prep(591, engine=engine, params='input/params.json')", number=1, globals=globals()), 'seconds to prep model.')
    # print(timeit("model.prep(591, engine=engine, params='input/params.json', attribute_multipliers='input/old_attribute_multipliers.json')", number=1, globals=globals()), 'seconds to prep model.')
    print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')

    # vals_json_attr = 'seir'
    # cmpts_json_attrs = ('age', 'vacc')
    # print(model.solution_sum([vals_json_attr] + list(cmpts_json_attrs)).stack(cmpts_json_attrs).index.droplevel('t').to_frame().to_dict(orient='records'))

    # model.write_to_db(engine, cmpts_json_attrs=('age', 'vacc', 'variant'))

    fig, ax = plt.subplots()
    modeled(model, 'Ih')
    actual_hosps(engine)
    plt.show()
