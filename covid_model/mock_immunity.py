import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import datetime as dt
import json
from time import perf_counter
from timeit import timeit

from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.model_with_omicron import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications


def two_cmpt_decay(t, scale1, scale2):
    in_cmpt1 = np.exp(-t / scale1)
    if scale1 == scale2:
        in_cmpt2 = t * np.exp(-t / scale1) / scale1
    else:
        in_cmpt2 = (scale2 * np.exp(-(t * (scale1 + scale2)) / (scale1 * scale2)) * (np.exp(t / scale2) - np.exp(t / scale1))) / (scale1 - scale2)

    return in_cmpt1 + in_cmpt2


# x = np.exp(-t / scale1)
# if scale1 != scale2:
#     y = (scale2 * np.exp(-(t * (scale1 + scale2)) / (scale1 * scale2)) * (np.exp(t / scale2) - np.exp(t / scale1))) / (scale1 - scale2)
#     z = (np.exp(-(t * (scale1 + scale2)) / (scale1 * scale2)) * (scale2 * np.exp(t / scale1) - np.exp(t / scale2) * (scale2 * np.exp(t / scale1) + scale1 * (-np.exp(t / scale1)) + scale1))) / (scale1 - scale2)
# else:
#     y = t * np.exp(-t / scale1) / scale1
#     z = (np.exp(-t / scale1) * (scale1 * np.exp(t / scale1) - scale1 - t)) / scale1


if __name__ == '__main__':
    engine = db_engine()

    spec_id = 532
    tmin = 30
    tmax = 30*18

    specs = CovidModelSpecifications.from_db(engine, 532)
    t = np.arange(tmin, tmax, 1.0)

    immunity_timeseries = dict()

    vacc_specs = specs.vacc_immun_params['shot2']
    initial_vacc_immunity = 1 - (pd.Series(vacc_specs['fail_rate']) * pd.Series(specs.model_params['group_pop']) / pd.Series(specs.model_params['group_pop']).sum()).sum()
    immunity_timeseries['Vaccine Immunity vs Alpha'] = initial_vacc_immunity * np.array([specs.vacc_eff_decay_mult(tx, 7, 1) for tx in t])
    immunity_timeseries['Vaccine Immunity vs Delta'] = initial_vacc_immunity * np.array([specs.vacc_eff_decay_mult(tx, 7, specs.model_params['delta_vacc_eff_k']) for tx in t])
    immunity_timeseries['Vaccine Immunity vs Omicron'] = initial_vacc_immunity * np.array([specs.vacc_eff_decay_mult(tx, 7, specs.model_params['omicron_vacc_eff_k']) for tx in t])

    initial_pinf_immunity = (specs.model_params['immune_rate_I'] + specs.model_params['immune_rate_A']) / (specs.model_params['lamb'] + 1)
    immunity_timeseries['Prior-Infection Immunity vs Alpha or Delta'] = initial_pinf_immunity * two_cmpt_decay(t, specs.model_params['immune_decay_days_1'], specs.model_params['immune_decay_days_2'])
    immunity_timeseries['Prior-Infection (non-Omicron) Immunity vs Omicron'] = specs.model_params['omicron_acq_immune_escape'] * immunity_timeseries['Prior-Infection Immunity vs Alpha or Delta']

    fig, ax = plt.subplots()
    colors = ['violet', 'mediumorchid', 'indigo', 'gold', 'orange']
    for (label, data), color in zip(immunity_timeseries.items(), colors):
        ax.plot(t, data, label=label, color=color)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(np.arange(tmin, tmax, 30))
    ax.set_xlabel('Days Since Infection or Vaccination')
    ax.set_ylabel('Infection Risk Reduction from Immunity')
    ax.grid(color='lightgray')
    ax.legend()

    plt.show()
