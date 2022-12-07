""" Python Standard Library """
import os
from time import perf_counter
import datetime as dt
from os.path import join

""" Third Party Imports """
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.cm import tab10, tab20, tab20b, tab20c

""" Local Imports """
from covid_model.model import CovidModel
from covid_model.utils import db_engine
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


def plot_modeled(model, compartments, ax=None, transform=lambda x: x, groupby=[], share_of_total=False, from_date=None,
                 **plot_params):
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
        ax.plot(model.daterange, model.solution_ydf.xs(g, level='group')[compartment],
                **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def plot_transmission_control(model, regions=None, **plot_params):
    # need to extend one more time period to see the last step. Assume it's the same gap as the second to last step
    tc_df = pd.DataFrame.from_dict(model.tc, orient='index').set_index(
        np.array([model.t_to_date(t) for t in model.tc.keys()]))
    if regions is not None:
        tc_df = tc_df[regions]  # regions should be a list
    tc_df.plot(drawstyle="steps-post", xlim=(model.start_date, model.end_date), **plot_params)


def format_date_axis(ax, interval_months=None, **locator_params):
    locator = mdates.MonthLocator(interval=interval_months) if interval_months is not None else mdates.AutoDateLocator(
        **locator_params)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(None)


def draw_stackplot(ax, df, f_title, l_title, ylabel, xlabel, colors):
    # Plot the dataframe values on the axis
    ax.stackplot(df.columns, df.values, labels=df.index, colors=colors)
    # Set up legend.
    ax.legend(title=l_title, fancybox=False, edgecolor="black", bbox_to_anchor=(1.0, 1.01), loc="upper left",
              fontsize=14, title_fontsize=14)
    # Set X-axis limit and label
    ax.set_xlim(df.columns.min(), df.columns.max())
    ax.set_xlabel(xlabel, fontsize=14)
    # Set Y-axis limit and label
    ax.set_ylabel(ylabel, fontsize=14)
    # Set title.
    ax.set_title(f_title, fontsize=16)


def plot_seir_comparments(df, fig_title, fig_filename=None, figsize=(14, 9)):
    seir = df.groupby(level="seir", axis=1).sum().T
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=seir,
                   f_title=fig_title,
                   l_title="Compartments",
                   xlabel="Time",
                   ylabel="Population",
                   colors=[tab10(i) for i in range(seir.index.nunique())])
    ax.set_ylim(0, seir.sum().max())
    plt.tight_layout()
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def plot_vacc_status_props(df, fig_title, fig_filename=None, figsize=(14, 9)):
    vacc_status = df.groupby(level=["vacc", "age"], axis=1).sum().T
    # We use gray for the none vaccination status, and colors otherwise
    colors = []
    color_groups = np.array([0, 1, 4, 2, 3])
    for vacc, cgrp in zip(vacc_status.index.get_level_values("vacc").unique(), color_groups):
        for j, age in enumerate(vacc_status.index.get_level_values("age").unique()):
            colors.append(tab20c((cgrp * 4) + j))
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=vacc_status,
                   f_title=fig_title,
                   l_title="Compartments",
                   xlabel="Time",
                   ylabel="Population",
                   colors=colors)
    ax.set_ylim(0, vacc_status.sum().max())
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def plot_variant_props(df, fig_title, fig_filename=None, figsize=(14, 9)):
    variants = df.drop(axis=1, level="seir", labels=["S", "E"]).groupby(level=["variant"], axis=1).sum().T
    variants = variants.divide(variants.sum(axis=0))
    variants.loc["none", variants.loc["none", :].isna()] = 1.0
    variants.fillna(0.0, inplace=True)
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=variants,
                   f_title=fig_title,
                   l_title="Compartments",
                   ylabel="Normalized Variant Proportion",
                   xlabel="Time",
                   colors=[tab10(i) for i in range(variants.index.nunique())])
    ax.set_ylim(0, 1)
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def plot_immunity_props(df, fig_title, fig_filename=None, figsize=(14, 9)):
    immun_df = df.groupby(level=["age", "immun"], axis=1).sum().T
    # reorder so that the colors make more intuitive sense.
    immun_df = immun_df.reindex(index=["strong", "weak", "none"], level=1)
    colors = []
    for i, age in enumerate(immun_df.index.get_level_values("age").unique()):
        for j, immun in enumerate(immun_df.index.get_level_values("immun").unique()):
            colors.append(tab20c((4 * i) + j))
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=immun_df,
                   f_title=fig_title,
                   l_title="Compartments",
                   xlabel="Time",
                   ylabel="Population",
                   colors=colors)
    ax.set_ylim(0, immun_df.sum().max())
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def generate_stackplots(model, output_dir=None, start_date=None, end_date=None):
    # Make a copy of the solution DF
    solution_df = model.solution_ydf.copy()

    # Set index to the actual dates for better readability
    solution_df.set_index(pd.date_range(model.start_date, periods=len(solution_df)), inplace=True)

    # Start and end dates can be specified, otherwise we use the full range of the model's solution
    start_date = model.start_date if start_date is None else start_date
    end_date = model.end_date if end_date is None else end_date
    solution_df = solution_df.loc[start_date:end_date]

    # Output directory defaults to current directory if not specified
    output_dir = os.getcwd() if output_dir is None else output_dir

    # Results figures and axes
    result_figs = {}

    # Plot SEIR
    seir_fig, seir_ax = plot_seir_comparments(df=solution_df,
                                              fig_title="SEIR Compartments",
                                              fig_filename=join(output_dir, "seir_status.png"))
    result_figs["seir"] = (seir_fig, seir_ax)

    # Plot Vaccination Status
    vacc_fig, vacc_ax = plot_vacc_status_props(df=solution_df,
                                               fig_title="Vaccination Status",
                                               fig_filename=join(output_dir, "vacc_status.png"))
    result_figs["vacc"] = (vacc_fig, vacc_ax)

    # Plot Variant Proportions
    var_fig, var_ax = plot_variant_props(df=solution_df,
                                         fig_title="Variant Distribution",
                                         fig_filename=join(output_dir, "variant_proportions.png"))
    result_figs["var"] = (var_fig, var_ax)

    # Plot Immunity Proportions
    immun_fig, immun_ax = plot_immunity_props(df=solution_df,
                                              fig_title="Immunity Status/Age",
                                              fig_filename=join(output_dir, "immunity_status.png"))
    result_figs["immun"] = (immun_fig, immun_ax)

    # Return results figures
    return result_figs
