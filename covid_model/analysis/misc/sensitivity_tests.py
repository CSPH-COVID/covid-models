import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import json
from covid_model.db import db_engine
from covid_model.model import CovidModelFit, CovidModel
from charts import modeled, actual_hosps, format_date_axis
from time import perf_counter
from timeit import timeit

if __name__ == '__main__':
    engine = db_engine()

    fig, ax = plt.subplots()

    param_fnames = ['input/params.json', 'input/params_with_new_imm_decay.json']
    fit_ids = [1409, 1368]
    colors = ['navy', 'green']

    for param_fname, fit_id, color in zip(param_fnames, fit_ids, colors):
        print(f'Running model with params from {param_fname} using fit {fit_id}...')
        model = CovidModel([0, 1000], engine=engine)
        model.set_ef_from_db(fit_id)
        model.add_tslice(800, 0.4)
        model.prep(params=param_fname)
        model.solve_seir()
        modeled(model, 'Ih', color=color)

    actual_hosps(engine)
    format_date_axis(ax)
    plt.show()
