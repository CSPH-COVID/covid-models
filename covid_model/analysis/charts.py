from covid_model.data_imports import get_deaths, get_hosps_df, get_hosps_by_age, get_deaths_by_age
from covid_model.model import CovidModel
from covid_model.db import db_engine
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib import cm, colors
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
# import pmdarima
import arch
import json
from time import perf_counter

from covid_model.data_imports import ExternalHosps


def actual_hosps(engine, county_ids=None, **plot_params):
    hosps = ExternalHosps(engine).fetch(county_ids=county_ids)['currently_hospitalized']
    hosps.plot(**{'color': 'red', 'label': 'Actual Hosps.', **plot_params})


def actual_hosps_by_group(engine, fname, axs, **plot_params):
    df = get_hosps_by_age(engine, fname)
    for g, ax in zip(CovidModel.groups, axs.flat):
        df.xs(g, level='group').plot(ax=ax, **{'label': f'Actual Hosps.', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def mmwr_infections_growth_rate(model: CovidModel):
    params_df = model.params_as_df
    new_infections = (model.solution_ydf.stack(level='age').stack(level='vacc')['E'] / params_df['alpha']).groupby('t').sum()
    growth_rate = 100 * (np.log(new_infections.cumsum()) - np.log(new_infections.shift(1).cumsum()))
    # growth_rate = 100 * (np.log(new_infections.rolling(150).sum()) - np.log(new_infections.shift(1).rolling(149).sum()))
    plt.plot(model.daterange[90:], growth_rate[90:])


def re_estimates(model: CovidModel):
    plt.plot(model.daterange[90:], model.re_estimates[90:])


def actual_deaths_by_group(engine, axs, **plot_params):
    df = get_deaths_by_age(engine)
    for g, ax in zip(CovidModel.groups, axs.flat):
        df.xs(g, level='group').plot(ax=ax, **{'label': f'Actual Deaths', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def actual_deaths(engine, **plot_params):
    deaths_df = get_deaths(engine)
    # deaths = list(deaths_df['cumu_deaths'])
    deaths_df['cumu_deaths'].plot(**{'color': 'red', 'label': 'Actual Deaths', **plot_params})


def actual_new_deaths(engine, rolling=1, **plot_params):
    deaths_df = get_deaths(engine)
    # deaths = list(deaths_df['new_deaths'].rolling(rolling).mean())
    deaths_df['new_deaths'].rolling(rolling).mean().plot(**{'color': 'red', 'label': 'Actual New Deaths', **plot_params})


def total_hosps(model, group=None, **plot_params):
    if group is None:
        hosps = model.total_hosps()
    else:
        hosps = model.solution_ydf.xs(group, level='group')['Ih']
    plt.plot(model.daterange, hosps, **{'c': 'blue', 'label': 'Modeled Hosps.', **plot_params})


def modeled(model, compartments, ax=None, transform=lambda x: x, groupby=[], share_of_total=False, **plot_params):
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
    df.plot(ax=ax, **plot_params)


def modeled_re(model, ax=None, **plot_params):
    if ax:
        ax.plot(model.daterange, model.re_estimates, **{**plot_params})
    else:
        plt.plot(model.daterange, model.re_estimates, **{**plot_params})


def modeled_by_group(model, axs, compartment='Ih', **plot_params):
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, model.solution_ydf.xs(g, level='group')[compartment], **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def transmission_control(model, **plot_params):
    plt.plot(model.fixed_tslices[:-1], model.tc, **plot_params)


def re_estimates(model, ax=None, **plot_params):
    if ax is not None:
        ax.plot(model.daterange[90:], model.re_estimates[90:], **plot_params)
    else:
        plt.plot(model.daterange[90:], model.re_estimates[90:], **plot_params)


def new_deaths_by_group(model, axs, **plot_params):
    deaths = model.solution_ydf['D'] - model.solution_ydf['D'].groupby('group').shift(1)
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, deaths.xs(g, level='group').shift(6), **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def total_deaths(model, **plot_params):
    modeled_deaths = model.solution_ydf_summed['D']
    modeled_deaths.index = model.daterange
    modeled_deaths.plot(**{'c': 'blue', 'label': 'Modeled Deaths.', **plot_params})


def new_deaths(model, **plot_params):
    modeled_deaths = model.solution_ydf_summed['D'] - model.solution_ydf_summed['D'].shift(1)
    modeled_deaths.index = model.daterange
    modeled_deaths.plot(**{'c': 'blue', 'label': 'Modeled New Deaths', **plot_params})


def actual_vs_modeled_hosps_by_group(actual_hosp_fname, model, **plot_params):
    fig, axs = plt.subplots(2, 2)
    actual_hosps_by_group(model.engine, actual_hosp_fname, axs=axs, c='red', **plot_params)
    modeled_by_group(model, axs=axs, compartment='Ih', c='blue', **plot_params)
    fig.tight_layout()


def actual_vs_modeled_deaths_by_group(engine, model, **plot_params):
    fig, axs = plt.subplots(2, 2)
    actual_deaths_by_group(engine, axs=axs, c='red', **plot_params)
    new_deaths_by_group(model, axs=axs, c='blue', **plot_params)
    fig.tight_layout()


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

# UQ histogram plot
# def uq_histogram(fit: CovidModelFit, sample_n=100, compartments='Ih', from_date=dt.datetime.now(), to_date=dt.datetime.now()+dt.timedelta(days=90)):
#     # get sample TC values
#     fitted_efs_dist = sps.multivariate_normal(mean=fit.fitted_efs, cov=fit.fitted_efs_cov)
#     samples = fitted_efs_dist.rvs(sample_n)
#
#     # for each sample, solve the model and add a line to the plot
#     model = CovidModel(fit.tslices, engine=engine)
#     model.add_tslice((to_date - model.start_date).days, 0)
#     model.prep()
#     for i, sample_fitted_efs in enumerate(samples):
#         print(i)
#         model.apply_tc(list(fit.fixed_efs) + list(sample_fitted_efs) + [sample_fitted_efs[-1] + tc_shift])
#         model.solve_seir()


def tc_for_given_r_and_vacc(solved_model: CovidModel, t, r, vacc_share):
    y = solved_model.solution_ydf_summed.loc[t]
    p = solved_model.gparams_lookup[t]
    current_vacc_immun = y['V'] / p[None]['N']
    acq_immun = (y['R'] + y['RA']) / p[None]['N'] * (1 + current_vacc_immun)  # adding back in the acq immun people who were moved to vacc immun
    eff_beta_at_tc100 = p[None]['beta'] * p[None]['rel_inf_prob'] * (y['I'] * p[None]['lamb'] + y['A'] * 1) / (y['I'] + y['A'])
    r0_at_tc100 = eff_beta_at_tc100 / p[None]['gamma']

    jnj_share = 0.078
    vacc_immun = (jnj_share*0.72 + (1-jnj_share)*0.9) * 0.9 * vacc_share
    r_at_tc100_space = r0_at_tc100 * (1 - vacc_immun - acq_immun + vacc_immun*acq_immun)
    return 1 - r / r_at_tc100_space


def r_for_given_tc_and_vacc(solved_model: CovidModel, t, tc, vacc_share):
    y = solved_model.solution_ydf_summed.loc[t]
    p = solved_model.gparams_lookup[t]
    current_vacc_immun = y['V'] / p[None]['N']
    acq_immun = (y['R'] + y['RA']) / p[None]['N'] * (1 + current_vacc_immun)  # adding back in the acq immun people who were moved to vacc immun
    # new_acq_immun_by_t = solved_model.solution_ydf_summed['E'] / 4.0 * 0.85 * np.exp(-(t - solved_model.solution_ydf_summed.index.values)/2514)
    # acq_immun = new_acq_immun_by_t.cumsum().loc[t] / p[None]['N']
    eff_beta_at_tc100 = p[None]['beta'] * p[None]['rel_inf_prob'] * (y['I'] * p[None]['lamb'] + y['A'] * 1) / (y['I'] + y['A'])
    r0_at_tc100 = eff_beta_at_tc100 / p[None]['gamma']

    jnj_share = 0.078
    vacc_immun = (jnj_share*0.72 + (1-jnj_share)*0.9) * 0.9 * vacc_share
    r_at_tc100_space = r0_at_tc100 * (1 - vacc_immun - acq_immun + vacc_immun*acq_immun)
    return r_at_tc100_space * (1 - tc)


def r_equals_1(solved_model: CovidModel, t=None):
    if t is None:
        t = (dt.datetime.now() - solved_model.start_date).days

    increment = 0.001
    vacc_space = np.arange(0.3, 0.95, increment)
    tc_space = np.arange(1.0, 0.4, -1 * increment)
    r_matrix = np.array([r_for_given_tc_and_vacc(solved_model, t, tc, vacc_space) for tc in tc_space])
    r_df = pd.DataFrame(data=r_matrix, index=['{:.0%}'.format(tc) for tc in tc_space], columns=['{:.0%}'.format(v) for v in vacc_space])
    # r_df = pd.DataFrame(data=r_matrix, index=tc_space, columns=vacc_immun_space)

    fig, ax = plt.subplots()
    sns.heatmap(r_df, ax=ax, cmap='Spectral_r', center=1.0, xticklabels=50, yticklabels=50, vmax=3.0, cbar_kws={"ticks": [0.0, 1.0, 2.0, 3.0]})
    fig.get_axes()[1].set_yticklabels(['Re = 0.0', 'Re = 1.0', 'Re = 2.0', 'Re = 3.0'])
    # sns.heatmap(r_df, ax=ax, cmap='bwr', center=1.0, xticklabels=50, yticklabels=50, vmax=3.0)

    # plot R = 1
    tc_for_r_equals_1_space = tc_for_given_r_and_vacc(solved_model, t, 1.0, vacc_space)
    ax.plot(range(len(vacc_space)), (tc_space.max() - tc_for_r_equals_1_space) // increment, color='navy')
    ax.set_xlabel('% of Population Fully Vaccinated')
    ax.set_ylabel('TCpb')
    ax.set_title('Re by Vaccination Rate and TCpb')
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # plot current point
    current_vacc = solved_model.vacc_rate_df.loc[t, 'cumu'].sum() / solved_model.gparams_lookup[t][None]['N']
    current_tc = solved_model.ef_by_t[t]
    current_r = r_for_given_tc_and_vacc(solved_model, t, current_tc, current_vacc)
    print(f"Current Vacc.: {'{:.0%}'.format(current_vacc)}")
    print(f"Current TCpb: {'{:.0%}'.format(current_tc)}")
    print(f"Current Re: {round(current_r, 2)}")
    ax.plot((current_vacc - vacc_space.min()) // increment, (tc_space.max() - current_tc) // increment, marker='o', color='navy')

    # formatting
    ax.grid(color='white')


def vaccination(fname='output/daily_vaccination_by_age (old).csv', **plot_params):
    df = pd.read_csv(fname, parse_dates=['measure_date']).set_index(['vacc_scen', 'group', 'measure_date'])
    first_shot_rate = df.loc['current trajectory', 'first_shot_rate'].groupby('measure_date').sum().rolling(7).mean()
    first_shot_rate.plot(**plot_params)


# def arima_garch_fit_and_sim(data, horizon=1):
#     transformed = np.log(1 - np.array(data))
#     # arima_model = pmdarima.auto_arima(transformed)
#     arima_model = pmdarima.ARIMA(order=(2, 0, 1), suppress_warnings=True).fit(transformed)
#
#     # fit ARIMA on transformed
#     p, d, q = arima_model.order
#     arima_residuals = arima_model.arima_res_.resid
#
#     # fit a GARCH(1,1) model on the residuals of the ARIMA model
#     garch = arch.arch_model(arima_residuals, p=1, q=1)
#     garch_model = garch.fit(disp='off')
#
#     # Use ARIMA to predict mu
#     predicted_mu = arima_model.predict(n_periods=horizon)
#
#     # Use GARCH to predict the residual
#     garch_forecast = garch_model.forecast(horizon=horizon, reindex=False, method='simulation')
#     # Combine both models' output: yt = mu + et
#     return 1 - np.exp(np.array([a + predicted_mu for a in garch_forecast.simulations.values[0]]))


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


def mab_projections(engine, fit_id):
    fig, ax = plt.subplots()

    print('Prepping model...')
    model = CovidModel([0, 800], engine=engine)
    model.set_ef_from_db(fit_id)
    print('Running model and building charts...')
    model.prep(params='input/params.json')
    model.solve_seir()
    modeled(model, 'Ih', c='royalblue', label='Current mAb uptake (14% of eligible)')

    for future_mab_uptake, color, label in zip([0.3, 0.5], ['green', 'deeppink'],
                                               ['Increase to 30% uptake over 3 weeks',
                                                'Increase to 50% uptake over 3 weeks']):
        current_mab_uptake = 0.53 * 0.14
        mab_eff_hosp = 0.78
        mab_eff_hlos = 0.28
        for t in range(650, 800):
            mab_uptake_at_t = current_mab_uptake + 0.53 * (0.5 - 0.14) * min(1., (t - 650) / (671 - 650))
            model.set_param('hosp', mult=(1 - mab_eff_hosp * mab_uptake_at_t) / (1 - mab_eff_hosp * current_mab_uptake),
                            trange=[t])
            model.set_param('hlos', mult=(1 - mab_eff_hlos * mab_uptake_at_t) / (1 - mab_eff_hlos * current_mab_uptake),
                            trange=[t])
        model.build_ode()
        model.solve_seir()
        modeled(model, 'Ih', c=color, label=label)

    actual_hosps(engine, color='navy', label='Actual')
    plt.legend(loc='best')
    locator = mdates.MonthLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylabel('Daily Patients Hospitalized with COVID-19')
    plt.grid(True)


if __name__ == '__main__':
    engine = db_engine()

    fig, ax = plt.subplots()
    print('Prepping model...')
    model = CovidModel([0, 800], engine=engine)
    model.set_ef_from_db(1263)
    print('Running model and building charts...')
    model.prep(params='input/params_with_increased_hosp_rate.json')
    t0 = perf_counter()
    model.solve_seir(method='LSODA')
    t1 = perf_counter()
    print(f'Solved ODE in {t1-t0} seconds.')
    modeled(model, 'Ih', c='royalblue', label='Current mAb uptake (14% of eligible)')

    # print('Prepping model...')
    # model = CovidModel([0, 800], engine=engine)
    # model.set_ef_from_db(1171)
    # print('Running model and building charts...')
    # model.prep(params='input/params_with_increased_hosp_rate.json')
    # model.solve_seir()
    # modeled(model, 'Ih', c='green')
    actual_hosps(engine)


    # fit = CovidModelFit.from_db(engine, 1263)
    # # fit = CovidModelFit.from_db(engine, 1163)
    # # fit.fitted_efs[-1] = fit.fitted_efs[-2]
    # # vacc_scens = ['current trajectory', 'increased booster uptake', '75% elig. boosters by Dec 31']
    # vacc_scens = ['current trajectory']
    # new_hosp_dists = pd.DataFrame()
    # for vacc_scen in vacc_scens:
    #     new_hosp_sims = plot_hosp_for_predicted_tcs(fit, total_sample_count=20, sample_tc_count=5
    #                                 , max_date=dt.datetime(2022, 5, 31)
    #                                 , engine=engine
    #                                 , model_params={'params': 'input/params_with_increased_hosp_rate.json',
    #                                 # , model_params={'params': 'input/params.json',
    #                                                          'vacc_proj_params': json.load(open('input/vacc_proj_params.json'))[vacc_scen]})
    #     new_hosp_dists[vacc_scen] = new_hosp_sims.loc[(dt.datetime.today() - dt.datetime(2020, 1, 24)).days:].sum()
    #
    # sns.displot(new_hosp_dists, kind='kde', fill=True)
    # print(new_hosp_dists.mean())
    # new_hosp_dists.to_csv('output/simulated_new_hosps_by_vacc_scen.csv')
    #
    plt.show()
    exit()

    # t0 = perf_counter()
    # model.solve_seir()
    # t1 = perf_counter()

    # print(f'Solved ODE in {t1 - t0} seconds.')
    from run_model_scenarios import build_legacy_output_df
    # model.write_to_db(engine)
    # modeled(model, 'Ih')
    # actual_hosps(engine)
    # plt.show()
    # exit()

    # actual_vs_modeled_hosps_by_group('input/hosps_by_group_20210611.csv', model)
    # plt.show()
    # exit()
    #
    # actual_vs_modeled_deaths_by_group(engine, model)
    # plt.show()

    # next_tc_sims = arima_garch_fit_and_sim(model.efs)[:10]

    # fig, ax = plt.subplots()
    # plt.grid(color='lightgray')
    # colors = ['tomato', 'royalblue', 'slategray']
    # fit = CovidModelFit.from_db(engine, 259)
    # # for i, tc_shift in enumerate([-0.10, 0.10, 0]):
    # #     uq_spaghetti(fit, sample_n=200, tmax=700, tc_shift=tc_shift, tc_shift_days=56, color=colors[i], alpha=0.05, compartments='Ih')
    # for next_tc in next_tc_sims:
    #     uq_spaghetti(fit, sample_n=200, tmax=707, tc_shift=next_tc - model.efs[0], tc_shift_days=0, alpha=0.05, compartments='Ih')
    # locator = mdates.MonthLocator(interval=2)
    # formatter = mdates.ConciseDateFormatter(locator)
    # ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.set_ylabel('Daily Patients Hospitalized with COVID-19')
    # # ax.set_ylabel('Cumulative Deaths from COVID-19')
    # plt.show()

    # uq_tc(CovidModelFit.from_db(engine, 4898), sample_n=10)

    # vaccination('output/daily_vaccination_by_age_old.csv', label='Old')
    # vaccination(label='New')
    # plt.legend(loc='best')
    # plt.show()

    # model.prep(vacc_proj_scen='current trajectory')
    # model.solve_seir()
    # r_equals_1(model)
    # plt.tight_layout()
    # plt.show()

    # model = CovidModel(params='input/params.json', tslices=[0, 700], engine=engine)
    # model.gparams.update({
    #   "N": 5840795,
    #   "groupN": {
    #     "0-19": 1513005,
    #     "20-39": 1685869,
    #     "40-64": 1902963,
    #     "65+": 738958
    #   }})
    # model.set_ef_from_db(1516)
    # model.set_ef_from_db(1644)
    # model.set_ef_from_db(1792)
    # model.efs[9] -= 0.07
    # model.efs[10] += 0.07
    # model.efs[11] += 0.01
    # model.set_ef_by_t(model.efs)
    # model.prep()
    # model.solve_seir()
    # actual_hosps(engine)
    # total_hosps(model)
    # actual_vs_modeled_hosps_by_group('input/hosps_by_group_20210611.csv', model)
    # actual_vs_modeled_deaths_by_group('input/deaths_by_group_20210614.csv', model)
    # plt.show()

    # fig, axs = plt.subplots(2, 2)
    # actual_deaths_by_group('input/deaths_by_group_20210614.csv', axs=axs)
    # plt.show()

    # actual_hosps(engine)
    # model = CovidModel.from_fit(engine, 1150)
    # model.prep()
    # model.solve_seir()
    # total_hosps(model)
    # plt.legend(loc='best')
    # plt.show()
    # exit()

    # actual_new_deaths(engine, rolling=7, label='Actual New Deaths (7-day avg.)')



    # new_deaths(model, c='royalblue', label='Current Parameters')
    #
    # model = CovidModel.from_fit(engine, 1150)
    # model.gparams['variants']['b117']['multipliers']['dh'] = 1.0
    # model.gparams['variants']['b117']['multipliers']['dnh'] = 1.0
    # model.prep()
    # model.solve_seir()
    # new_deaths(model, c='orange', label='With No B117 Impact on Death')
    #
    # model = CovidModel.from_fit(engine, 1150)
    # for vacc in ['mrna', 'jnj']:
    #     for shot in ['first_shot', 'second_shot']:
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dh'] = 0.4
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dnh'] = 0.2
    # model.prep()
    # model.solve_seir()
    # new_deaths(model, c='lightseagreen', label='With 90% Vacc. Protection vs. Death (instead of 75%)')
    #
    # model = CovidModel.from_fit(engine, 1150)
    # model.gparams['variants']['b117']['multipliers']['dh'] = 1.0
    # model.gparams['variants']['b117']['multipliers']['dnh'] = 1.0
    # for vacc in ['mrna', 'jnj']:
    #     for shot in ['first_shot', 'second_shot']:
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dh'] = 0.4
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dnh'] = 0.2
    # model.prep()
    # model.solve_seir()
    # new_deaths(model, c='tomato', label='With Both Vacc & Variant Adjust.')

    # model = CovidModel.from_fit(engine, 1150)
    # for vacc in ['mrna', 'jnj']:
    #     for shot in ['first_shot', 'second_shot']:
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dh'] = 0.4
    #         model.gparams['vaccines'][vacc]['shots'][shot]['multipliers']['dnh'] = 0.2
    # model.gparams['dh']['0-19'] *= 0.1
    # model.gparams['dnh']['0-19'] *= 0.1
    # model.gparams['dh']['20-39'] *= 0.1
    # model.gparams['dnh']['20-39'] *= 0.1
    # model.gparams['dh']['40-64'] *= 0.1
    # model.gparams['dnh']['40-64'] *= 0.1
    # model.gparams['dh']['65+'] *= 1.12
    # model.gparams['dnh']['65+'] *= 1.12
    # model.prep()
    # model.solve_seir()
    # total_deaths(model, c='tab:purple', label='With Under-40 Death-Rate Reduced by 20%')

    # actual_hosps(engine)
    # model = CovidModel('input/params.json', [0, 700], engine=engine)
    # model.set_ef_from_db(1992)
    # model.prep(vacc_proj_scen='current trajectory')
    # model.solve_seir()
    # total_hosps(model)
    # uq_spaghetti(CovidModelFit.from_db(engine, 2330), sample_n=200, tmax=600)
    # uq_spaghetti(CovidModelFit.from_db(engine, 1992), sample_n=200, tmax=600)

    # actual_hosps(engine, color='black')
    # model = CovidModel('input/params.json', [0, 700], engine=engine)
    # model.set_ef_from_db(3472)
    # model.prep(vacc_proj_scen='current trajectory')
    # model.solve_seir()
    # modeled(model, ['Ih'], color='red')
    #
    # plt.show()

    # fig, axs = plt.subplots(2, 2)

    # fits = {'6-12 mo. immunity': 2470, '12-24 mo. immunity': 2467, 'indefinite immunity': 2502}
    # colors = ['r', 'b', 'g', 'black', 'orange', 'pink']
    # for i, (label, fit_id) in enumerate(fits.items()):
    #     fit1 = CovidModelFit.from_db(conn=engine, fit_id=fit_id)
    #     model1 = CovidModel(fit1.model_params, [0, 600], engine=engine)
    #     model1.set_ef_from_db(fit_id)
    #     model1.prep()
    #     model1.solve_seir()
    #     # modeled(model1, compartments=['E'], transform=lambda x: x.cumsum()/4.0, c=colors[i], label=label)
    #     # modeled(model1, compartments=['V', 'Vxd'], transform=lambda x: x/5813208.0, c=colors[i], label=label)
    #     modeled(model1, compartments=['I', 'A'], c=colors[i], label=label)
    #     # transmission_control(model1, c=colors[i], label=label)
    #     # re_estimates(model1, c=colors[i], label=label)
    #     # modeled_by_group(model1, axs=axs, compartments=['I', 'A'], c=colors[i], label=label)
    #
    # plt.legend(loc='best')
    # plt.ylabel('People Infected')
    # plt.show()

    # plt.legend(loc='best')
    # plt.xlabel('Days')
    # # plt.xlim('2021-02-01', '2021-05-17')
    # # plt.ylim(0, 25)
    # plt.grid()
    # plt.show()


