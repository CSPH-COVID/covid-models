import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import json
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates
from time import perf_counter
from timeit import timeit

from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.model_specs import CovidModelSpecifications

if __name__ == '__main__':
    engine = db_engine()

    model = CovidModel()
    cms = CovidModelSpecifications.from_db(engine, 325, new_end_date=model.end_date)
    cms.set_model_params('input/params.json')

    model.prep(specs=cms)

    # model.set_tc(tc)
    # model.prep(tc=tc)

    fig, ax = plt.subplots()
    print('Prepping model...')
    print(timeit('model.prep()', number=1, globals=globals()), 'seconds to prep model.')

    # for method in ['RK45', 'RK23', 'BDF', 'LSODA', 'Radau', 'DOP853']:
    for method in ['RK45']:
        print(timeit('model.solve_seir(method=method)', number=1, globals=globals()), 'seconds to run model.')

    # model.write_to_db(engine)

    modeled(model, 'Ih')
    actual_hosps(engine)
    plt.show()
