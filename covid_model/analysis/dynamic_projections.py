import numpy as np
import pandas as pd
import datetime as dt
import json
import random

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns

import scipy.stats as sps
import pmdarima
import arch

from covid_model.db import db_engine
from covid_model.model import CovidModel
from charts import plot_kde, modeled, actual_hosps, format_date_axis


def uq_sample_tcs(fit, sample_n):
    fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_tc, cov=fit.fitted_efs_cov)
    fitted_efs_samples = fitted_efs_dist.rvs(sample_n)
    return [list(fit.fixed_tc) + list(sample) for sample in (fitted_efs_samples if sample_n > 1 else [fitted_efs_samples])]


def arima_garch_fit_and_sim(data, horizon=1, sims=10, arima_order='auto', use_garch=False):
    historical_values = np.log(1 - np.array(data))
    if arima_order is None or arima_order == 'auto':
       arima_model = pmdarima.auto_arima(historical_values, suppress_warnings=True, seasonal=False)
    else:
        arima_model = pmdarima.ARIMA(order=arima_order, suppress_warnings=True).fit(historical_values)
    arima_results = arima_model.arima_res_
    p, d, q = arima_model.order

    # fit ARIMA on transformed
    arima_residuals = arima_model.arima_res_.resid

    if use_garch:
        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch = arch.arch_model(arima_residuals, p=1, q=1)
        garch_model = garch.fit(disp='off')
        garch_sims = [e[0] for e in garch_model.forecast(horizon=1, reindex=False, method='simulation').simulations.values[0]]

        # simulate projections iteratively
        all_projections = []
        for i in range(sims):
            projections = []
            for steps_forward in range(horizon):
                projected_error = random.choice(garch_sims)
                projected_mean = arima_results.forecast(1)[0]

                projections.append(projected_mean + projected_error)
                arima_results = arima_results.append([projections[-1]])

            all_projections.append(projections)

    else:
        projections = arima_results.simulate(horizon, anchor='end', repetitions=sims)
        all_projections = [[projections[step][0][rep] for step in range(horizon)] for rep in range(sims)]

    return 1 - np.exp(np.array(all_projections))


def plot_spaghetti(data, ax, ylabel=None):
    alpha = min(1., 10. / len(data.columns))
    ax.plot(data, color='navy', alpha=alpha)

    format_date_axis(ax)
    ax.set_ylabel(ylabel)


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


def plot_peak_distribution(data, ax, xlabel='Peak'):
    peak_value = data.max(axis=0)
    plot_kde(peak_value, ax=ax, color='navy')
    ax.set_xlabel(xlabel)


def plot_peak_value_vs_date(data, ax, xlabel='Date of Peak', ylabel='Peak Value'):
    peak_value = data.max(axis=0)
    peak_date = data.idxmax(axis=0)
    ax.scatter(peak_date, peak_value, color='darkgreen', alpha=0.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_date_axis(ax)


def plot_date_of_peak(data, ax, xlabel='Date of Peak'):
    peak_date = data.iloc[fit.fixed_tslices[-1]:].idxmax(axis=0)
    plot_kde(peak_date, ax=ax, color='darkgreen')
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel(xlabel)


def get_tc_sims(fit, max_date, sample_count, samples_per_fit_sim=5, plot=False, arima_order='auto', skip=8):
    increment = 14
    max_t = (max_date - dt.datetime(2020, 1, 24)).days
    tslices = fit.fixed_tslices + list(range(fit.fixed_tslices[-1] + increment, max_t, 14)) + [max_t]

    sample_tcs = uq_sample_tcs(fit, sample_count // samples_per_fit_sim)
    horizon = len(tslices) - 1 - len(sample_tcs[0])

    tc_sims = []
    i = 0
    for tcs in sample_tcs:
        print(f'Generating simulated TCs #{i}-{i+samples_per_fit_sim-1}...')
        next_tcs = arima_garch_fit_and_sim(tcs[skip:], horizon=horizon, arima_order=arima_order)
        for next_tc in next_tcs[:samples_per_fit_sim]:
            i += 1
            tc_sims.append(list(tcs) + list(next_tc))

    tc_sims = pd.DataFrame(tc_sims).transpose()
    tc_sims.index = [dt.date(2020, 1, 24) + dt.timedelta(days=t) for t in tslices[:-1]]

    if plot:
        fig, ax = plt.subplots()
        plot_prediction_interval(tc_sims, ax=ax, ylabel='Projected Transmission Control', color='darkorange',
                                 ci=0.50, alpha=0.5)
        plot_prediction_interval(tc_sims, ax=ax, ylabel='Projected Transmission Control', color='darkorange',
                                 ci=0.95, alpha=0.3)


def get_sims(fit: CovidModelFit, engine, max_date, sample_count, samples_per_fit_sim=5, output_dir='output', arima_order='auto', skip=8, model_params={}):
    increment = 14
    max_t = (max_date - dt.datetime(2020, 1, 24)).days
    # fit.tslices[-1] += increment // 2
    tslices = fit.tslices + list(range(fit.tslices[-1] + increment, max_t, 14)) + [max_t]

    sample_tcs = uq_sample_tcs(fit, sample_count//samples_per_fit_sim)
    horizon = len(tslices) - 1 - len(sample_tcs[0])

    model = CovidModel(tslices=tslices, engine=engine)
    model.prep(**model_params)
    params_df = model.params_as_df

    hosp_sims = {}
    new_infection_sims = {}
    new_hosp_sims = {}
    death_sims = {}
    tc_sims = {}
    i = 0
    for tcs in sample_tcs:
        next_tcs = arima_garch_fit_and_sim(tcs[skip:], horizon=horizon, arima_order=arima_order)
        print(f'Generating predictions for next TC for sampled fitted TCs: {tcs}')
        for next_tc in next_tcs[:min(samples_per_fit_sim, sample_count)]:
            i += 1
            print(f'Generating and plotting simulation number {i}...')
            model.apply_tc(list(tcs) + list(next_tc))
            model.solve_seir()
            tc_sims[i] = model.ef_by_t
            hosp_sims[i] = model.solution_sum('seir')['Ih']
            new_infection_sims[i] = (model.solution_ydf.stack(level='age').stack(level='vacc')['E'] / params_df['alpha']).groupby('t').sum()
            new_hosp_sims[i] = (model.solution_ydf.stack(level='age').stack(level='vacc')['I'] * params_df['gamm'] * params_df['hosp']).groupby('t').sum()
            death_sims[i] = model.solution_sum('seir')['D']
            # modeled(model, ax=ax, compartments='Ih', **{'c': 'navy', 'alpha': min(1, 10.0/total_sample_count), **plot_params})

    tc_sims = pd.DataFrame(tc_sims)
    tc_sims.index = model.daterange
    tc_sims.to_csv(f'{output_dir}/tc_sims.csv', header=False)

    hosp_sims = pd.DataFrame(hosp_sims)
    hosp_sims.index = model.daterange
    hosp_sims.to_csv(f'{output_dir}/census_hosps.csv', header=False)

    new_infection_sims = pd.DataFrame(new_infection_sims)
    new_infection_sims.index = model.daterange
    new_infection_sims.to_csv(f'{output_dir}/new_infections.csv', header=False)

    new_hosp_sims = pd.DataFrame(new_hosp_sims)
    new_hosp_sims.index = model.daterange
    new_hosp_sims.to_csv(f'{output_dir}/new_hosps.csv', header=False)

    death_sims = pd.DataFrame(death_sims)
    death_sims.index = model.daterange
    death_sims.to_csv(f'{output_dir}/total_deaths.csv', header=False)

    return tc_sims, hosp_sims, new_infection_sims, new_hosp_sims, death_sims


def plot_projection_summary(fit: CovidModelFit, engine, max_date, sample_count, model_params={}, **plot_params):
    fig, axs = plt.subplots(3, 3, figsize=(16, 11))
    plt.grid(color='lightgray')

    tc_sims, hosp_sims, new_hosp_sims, death_sims = get_sims(fit, engine, max_date, sample_count, model_params=model_params)

    start_date = dt.datetime.today().strftime('%Y-%m-%d')
    end_date_str = max_date.strftime("%b %#d, %Y")
    # plot_spaghetti(hosp_sims, ax=axs[0, 0], ylabel='Daily Patients Hospitalized with COVID-19')
    plot_prediction_interval(hosp_sims, ax=axs[0, 0], ylabel='Daily Patients Hospitalized with COVID-19', ci=0.50, alpha=0.5)
    plot_prediction_interval(hosp_sims, ax=axs[0, 0], ylabel='Daily Patients Hospitalized with COVID-19', ci=0.95, alpha=0.3)
    actual_hosps(engine, ax=axs[0, 0], color='xkcd:midnight')
    plot_prediction_interval(tc_sims, ax=axs[1, 0], ylabel='Projected Transmission Control', color='darkorange', ci=0.50, alpha=0.5)
    plot_prediction_interval(tc_sims, ax=axs[1, 0], ylabel='Projected Transmission Control', color='darkorange', ci=0.95, alpha=0.3)
    plot_peak_distribution(hosp_sims.loc[start_date:], ax=axs[0, 1], xlabel=f'Hospitalized Patients Peak From Now Through {end_date_str}')
    plot_peak_value_vs_date(hosp_sims.loc[start_date:], ax=axs[0, 2])
    # plot_date_of_peak(hosp_sims, ax=axs[0, 2], xlabel=f'Day of Hospitalizations Peak From Now Through {end_date_str}')
    plot_kde(new_hosp_sims.loc[start_date:].sum(axis=0), ax=axs[1, 1], color='royalblue', xlabel=f'Total Hospital Admissions From Now Through {end_date_str}')
    plot_kde(death_sims.iloc[-1] - death_sims.loc[start_date], ax=axs[1, 2], color='xkcd:charcoal', xlabel=f'Total Deaths From Now Through {end_date_str}')
    for i, days in enumerate([14, 28, 56]):
        d = dt.datetime.today() + dt.timedelta(days=days)
        plot_kde(hosp_sims.loc[d.strftime('%Y-%m-%d')], ax=axs[2, i], color='teal', xlabel=f'Patients Hospitalized with COVID-19 on {d.strftime("%b %#d, %Y")}')

    [ax.grid(True, color='lightgray') for ax in axs.flatten()]
    fig.tight_layout(pad=3)


def plot_projection_comparison(engine, fits, model_params_dict, sample_count=20, max_date=dt.datetime(2022, 5, 31), base_output_dir=None):
    fig, ax = plt.subplots(figsize=(11, 11))
    # plt.grid(color='lightgray')

    if len(fits) == 1:
        fits = fits * len(model_params_dict.keys())
    colors = ['royalblue', 'green', 'orange']
    for fit, (scen_label, model_params), color in zip(fits, model_params_dict.items(), colors):
        output_dir = None
        if base_output_dir is not None:
            output_dir = f'{base_output_dir}/{scen_label}'
        tc_sims, hosp_sims, new_hosp_sims, death_sims = get_sims(fit, engine, max_date=max_date, sample_count=sample_count, model_params={'vacc_proj_params': model_params}, output_dir=output_dir)
        start_date = dt.datetime.today().strftime('%Y-%m-%d')
        end_date_str = max_date.strftime("%b %#d, %Y")
        plot_kde(new_hosp_sims.loc[start_date:].sum(axis=0), ax=ax, ci=None, xlabel=f'Total Hospital Admissions From Now Through {end_date_str}', label=scen_label, color=color)
        plt.legend(loc='best')


if __name__ == '__main__':
    engine = db_engine()
    fit = CovidModelFit.from_db(engine, 1735)

    # plot_projection_summary(fit, sample_count=400
    #                         , max_date=dt.datetime(2022, 5, 31)
    #                         , engine=engine
    #                         , model_params={'params': 'input/params_with_new_imm_decay.json'})

    # params = json.load(open('input/vacc_proj_params.json'))
    # del params["current trajectory w/o 5-11 vacc."]
    # plot_projection_comparison(engine, fits=[fit], model_params_dict=params, sample_count=100, base_output_dir='output/sims/vacc_projs')


    get_tc_sims(fit, sample_count=500, samples_per_fit_sim=10, max_date=dt.datetime(2022, 5, 31), plot=True, arima_order=(2, 0, 1))
    plt.show()




    # vacc_scens = ['current trajectory']
    # new_hosp_dists = pd.DataFrame()
    # for vacc_scen in vacc_scens:
    #     new_hosp_sims = plot_hosp_for_predicted_tcs(fit, total_sample_count=20, sample_tc_count=5
    #                                                 , max_date=dt.datetime(2022, 5, 31)
    #                                                 , engine=engine
    #                                                 , model_params={'params': 'input/params_with_increased_hosp_rate.json',
    #                                                                 # , model_params={'params': 'input/params.json',
    #                                                                 'vacc_proj_params':
    #                                                                     json.load(open('input/vacc_proj_params.json'))[
    #                                                                         vacc_scen]})
    #     new_hosp_dists[vacc_scen] = new_hosp_sims.loc[(dt.datetime.today() - dt.datetime(2020, 1, 24)).days:].sum()
    #
    # sns.displot(new_hosp_dists, kind='kde', fill=True)
    # print(new_hosp_dists.mean())
    # new_hosp_dists.to_csv('output/simulated_new_hosps_by_vacc_scen.csv')

