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
from covid_model.data_imports import ExternalHosps


def actual_hosps(engine, county_ids=None, **plot_params):
    hosps = ExternalHosps(engine).fetch(county_ids=county_ids)['currently_hospitalized']
    hosps.index = pd.to_datetime(hosps.index)
    hosps.plot(**{'color': 'red', 'label': 'Actual Hosps.', **plot_params})


def modeled(model, compartments, ax=None, transform=lambda x: x, groupby=[], share_of_total=False, from_date=None, **plot_params):
    if type(compartments) == str:
        compartments = [compartments]

    if groupby:
        if type(groupby) == str:
            groupby = [groupby]
        df = transform(model.solution_sum(['seir', *groupby])[compartments].groupby(groupby, axis=1).sum())
        if share_of_total:
            total = df.sum(axis=1)
            df = df.apply(lambda s: s / total)
    else:
        df = transform(model.solution_sum('seir'))
        if share_of_total:
            total = df.sum(axis=1)
            df = df.apply(lambda s: s / total)
        df = df[compartments].sum(axis=1)

    df.index = model.daterange
    if from_date is not None:
        df = df.loc[from_date]

    df.plot(ax=ax, **plot_params)

    if share_of_total:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def re_estimates(model, ax=None, **plot_params):
    if ax is not None:
        ax.plot(model.daterange[90:], model.re_estimates[90:], **plot_params)
    else:
        plt.plot(model.daterange[90:], model.re_estimates[90:], **plot_params)


def modeled_by_group(model, axs, compartment='Ih', **plot_params):
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, model.solution_ydf.xs(g, level='group')[compartment], **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def transmission_control(model, **plot_params):
    # need to extend one more time period to see the last step. Assume it's the same gap as the second to last step
    tcs = model.tc + [np.infty]  # the last tc doesn't get plotted b/c of "steps-post" style.
    tend = [2*model.tslices[-1]-model.tslices[-2]] if len(model.tslices) > 1 else [2*model.tslices[0]]
    tslices = [0] + model.tslices + tend # 0 is implicit first tslice
    tc_df = pd.DataFrame(tcs, columns=['TC'], index=[model.start_date + dt.timedelta(days=d) for d in tslices])
    tc_df.plot(drawstyle="steps-post", **plot_params)


# UQ TC plot
# def uq_tc(fit: CovidModelFit, sample_n=100, **plot_params):
#     # get sample TC values
#     fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_efs, cov=fit.fitted_efs_cov)
#     samples = fitted_efs_dist.rvs(sample_n)
#     for sample_fitted_efs in samples:
#         plt.plot(fit.tslices[:-1], list(fit.fixed_efs) + list(sample_fitted_efs), **{'marker': 'o', 'linestyle': 'None', 'color': 'darkorange', 'alpha': 0.025, **plot_params})
#
#     plt.xlabel('t')
#     plt.ylabel('TCpb')


def uq_sample_tcs(fit, sample_n):
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_tc, cov=fit.fitted_efs_cov)
    fitted_efs_samples = fitted_efs_dist.rvs(sample_n)
    return [list(fit.fixed_tc) + list(sample) for sample in (fitted_efs_samples if sample_n > 1 else [fitted_efs_samples])]


# UQ sqaghetti plot
def uq_spaghetti(fit, sample_n=100, tmax=600,
                 tc_shift=0, tc_shift_days=70, tc_shift_date=dt.date.today() + dt.timedelta(2) + dt.timedelta((6-dt.date.today().weekday()) % 7),
                 compartments='Ih', **plot_params):
    # get sample TC values
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_tc, cov=fit.fitted_efs_cov)
    samples = fitted_efs_dist.rvs(sample_n)

    # for each sample, solve the model and add a line to the plot
    model = CovidModel(fit.fixed_tslices + [tmax], list(fit.fixed_tc) + list(samples[0]) + [0], engine=engine)
    # model.set_ef_from_db(fit.fit_id)
    # model.add_tslice(tmax, 0)
    model.prep()
    for i, sample_fitted_efs in enumerate(samples):
        print(i)
        # model = base_model.prepped_duplicate()
        current_ef = sample_fitted_efs[-1]
        if tc_shift_days is None:
            model.apply_tc(list(fit.fixed_tc) + list(sample_fitted_efs) + [current_ef + tc_shift])
        else:
            model.apply_tc(list(fit.fixed_tc) + list(sample_fitted_efs) + [current_ef])
            for i, tc_shift_for_this_day in enumerate(np.linspace(0, tc_shift, tc_shift_days)):
                model.ef_by_t[(tc_shift_date - model.start_date.date()).days + i] += tc_shift_for_this_day
            for t in range((tc_shift_date - model.start_date.date()).days + tc_shift_days, tmax):
                model.ef_by_t[t] += tc_shift
        # base_model.set_ef_by_t(list(fit.fixed_efs) + list(sample_fitted_efs) + [sample_fitted_efs[-1] + tc_shift])
        model.solve_seir()
        modeled(model, compartments=compartments, **{'c': 'darkblue', 'alpha': 0.025, **plot_params})
        # plt.plot(model.daterange, model.total_hosps(), **{'color': 'darkblue', 'alpha': 0.025, **plot_params})


def format_date_axis(ax, interval_months=None, **locator_params):
    locator = mdates.MonthLocator(interval=interval_months) if interval_months is not None else mdates.AutoDateLocator(**locator_params)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(None)


def plot_kde(data, ax, fill=True, xlabel=None, ci=0.8, **plot_params):
    # if 'color' in plot_params.keys():
    #     plot_params['colors'] = plot_params['color']
    p = sns.kdeplot(data, ax=ax, fill=False, **plot_params)
    if len(p.get_lines()) > 0:
        x, y = p.get_lines()[-1].get_data()
        try:
            if ci is not None:
                for percentile in [0.5 - ci/2, 0.5, 0.5 + ci/2]:
                    median_i = np.argmax(x > data.quantile(percentile))
                    ax.vlines(x[median_i], 0, y[median_i], linestyles='dashed', **plot_params)

            mean_i = np.argmax(x > data.mean())
            ax.vlines(x[mean_i], 0, y[mean_i], linestyles='dotted', **plot_params)
        except:
            print('Oops!')

    sns.kdeplot(data, ax=ax, fill=fill, **plot_params)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(None)
    ax.axes.yaxis.set_ticks([])