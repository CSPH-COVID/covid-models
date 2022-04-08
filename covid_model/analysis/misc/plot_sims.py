### Python Standard Library ###
import datetime as dt
### Third Party Imports ###
import matplotlib.pyplot as plt
### Local Imports ###
from covid_model.model_sims import CovidModelSimulation
from covid_model.db import db_engine
from covid_model.analysis.charts import format_date_axis


def plot_prediction_interval(data, ax, ci=0.8, ylabel=None, save_fig=None, format_as_dates=True, **plot_params):
    hosp_low = data.quantile(0.5 - ci/2, axis=1)
    hosp_mid = data.quantile(0.5, axis=1)
    hosp_high = data.quantile(0.5 + ci/2, axis=1)
    # hosp_mid.plot(ax=ax, c='navy')
    ax.fill_between(hosp_low.index, hosp_low, hosp_high, **{'color': 'navy', 'alpha': 0.3, **plot_params})

    if format_as_dates:
        format_date_axis(ax)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if save_fig:
        plt.savefig(save_fig)


def format_and_save_plot(ax, fig, fpath):
    ax.legend(loc='upper left')
    ax.set_xlim(dt.datetime(2020, 3, 1), dt.datetime(2022, 3, 31))
    ax.set_ylim(0, 2500)
    ax.set_ylabel('Hospitalized with COVID-19')
    fig.savefig(fpath, bbox_inches='tight')

if __name__ == '__main__':
    engine = db_engine()

    # base projection
    sim = CovidModelSimulation.from_db(engine, 92)
    fig, ax = plt.subplots()
    sim.results_hosps.median(axis=1).plot(ax=ax, color='navy', label='Median')
    plot_prediction_interval(sim.results_hosps, ax=ax, color='navy', label='50% Prediction Interval', ci=0.50, alpha=0.6)
    plot_prediction_interval(sim.results_hosps, ax=ax, color='navy', label='90% Prediction Interval', ci=0.90, alpha=0.3)
    format_and_save_plot(ax, fig, 'output/base_projection.png')

    # base projection
    mab_sim = CovidModelSimulation.from_db(engine, 110)
    fig, ax = plt.subplots()
    plot_prediction_interval(sim.results_hosps, ax=ax, color='royalblue', label='Current trajectory', ci=0.50, alpha=0.6)
    plot_prediction_interval(mab_sim.results_hosps, ax=ax, color='firebrick', label='Increased mAb Uptake (50% of eligible)', ci=0.50, alpha=0.6)
    # plot_prediction_interval(mab_sim.results_hosps, ax=ax, color='firebrick', label='Increased booster uptake', ci=0.50, alpha=0.6)
    format_and_save_plot(ax, fig, 'output/mab_projection.png')

