### Python Standard Library ###
from time import perf_counter
import datetime as dt
### Third Party Imports ###
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
### Local Imports ###
from covid_model.model import CovidModel
from covid_model.db import db_engine
from covid_model.data_imports import ExternalHospsEMR, ExternalHospsCOPHS


def plot_observed_hosps(engine, county_ids=None, **plot_params):
    # TODO: pass in model and use its hosps instead of getting from db
    if county_ids is None:
        hosps = ExternalHospsEMR(engine).fetch()['currently_hospitalized']
    else:
        hosps = ExternalHospsCOPHS(engine).fetch(county_ids=county_ids)['currently_hospitalized']
    hosps.plot(**{'color': 'red', 'label': 'Actual Hosps.', **plot_params})


def plot_modeled_vs_actual_hosps():
    # TODO
    pass


def plot_modeled(model, compartments, ax=None, transform=lambda x: x, groupby=[], share_of_total=False, from_date=None, **plot_params):
    if type(compartments) == str:
        compartments = [compartments]

    if groupby:
        if type(groupby) == str:
            groupby = [groupby]
        df = transform(model.solution_sum_df(['seir', *groupby])[compartments].groupby(groupby, axis=1).sum())
        if share_of_total:
            total = df.sum(axis=1)
            df = df.apply(lambda s: s / total)
    else:
        df = transform(model.solution_sum_df('seir'))
        if share_of_total:
            total = df.sum(axis=1)
            df = df.apply(lambda s: s / total)
        df = df[compartments].sum(axis=1)

    if from_date is not None:
        df = df.loc[from_date]

    df.plot(ax=ax, **plot_params)

    if share_of_total:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def plot_modeled_by_group(model, axs, compartment='Ih', **plot_params):
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, model.solution_ydf.xs(g, level='group')[compartment], **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def plot_transmission_control(model, **plot_params):
    # need to extend one more time period to see the last step. Assume it's the same gap as the second to last step
    tc_df = pd.DataFrame.from_dict(model.tc, orient='index').set_index(np.array([model.t_to_date(t) for t in model.tc.keys()]))
    tc_df.plot(drawstyle="steps-post", xlim=(model.start_date, model.end_date), **plot_params)


def format_date_axis(ax, interval_months=None, **locator_params):
    locator = mdates.MonthLocator(interval=interval_months) if interval_months is not None else mdates.AutoDateLocator(**locator_params)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(None)