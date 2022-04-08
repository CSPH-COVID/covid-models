### Python Standard Library ###
import datetime as dt
import json
import os
from pathlib import Path
### Third Party Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
### Local Imports ###
from covid_model.analysis.misc.dynamic_projections import get_sims, plot_prediction_interval
from covid_model.db import db_engine
from covid_model.model_fit import CovidModelFit


def build_mab_prevalence(base_mab_prevalence, mab_prevalence_target, mab_prevalence_target_date, start_date=dt.datetime.today().strftime('%Y-%m-%d')):
    increase_mab_prevalence_over_dates = pd.date_range(start_date, mab_prevalence_target_date)
    mab_prevalence = base_mab_prevalence.copy()
    mab_prevalence.loc[start_date:mab_prevalence_target_date, 'treatment_mab'] = np.linspace(mab_prevalence.loc[start_date, 'treatment_mab'],
                                                                        mab_prevalence_target,
                                                                        len(increase_mab_prevalence_over_dates))
    mab_prevalence.loc[mab_prevalence_target_date:, 'treatment_mab'] = mab_prevalence_target
    return mab_prevalence


def get_sims_from_files(dirname):
    census_hosps = pd.read_csv(os.path.join(dirname, 'census_hosps.csv'), parse_dates=[0], index_col=0, header=None)
    new_infections = pd.read_csv(os.path.join(dirname, 'new_infections.csv'), parse_dates=[0], index_col=0, header=None)
    new_hosps = pd.read_csv(os.path.join(dirname, 'new_hosps.csv'), parse_dates=[0], index_col=0, header=None)
    total_deaths = pd.read_csv(os.path.join(dirname, 'total_deaths.csv'), parse_dates=[0], index_col=0, header=None)
    tc_sims = pd.read_csv(os.path.join(dirname, 'tc_sims.csv'), parse_dates=[0], index_col=0, header=None)

    return tc_sims, census_hosps, new_infections, new_hosps, total_deaths


if __name__ == '__main__':
    data_dir = 'output/sims/vacc_projs'
    start_date = dt.datetime.today().strftime('%Y-%m-%d')
    engine = db_engine()
    fit = CovidModelFit.from_db(engine, 1907)
    base_raw_params = json.load(open('input/params.json'))
    get_from_files = True

    # pov_scens = {'no proof of vacc. req.': 1.0, 'proof of vacc. req. on Nov. 12': 0.93}
    # mask_scens = {'no mask req.': 1.0, 'mask req. on Nov. 12': 0.897, 'mask req. Nov 12 - Jan 7': 0.897, 'proof of vacc. req. on Nov. 12': 0.85}
    # mask_scens = {'no mask req.': 1.0, 'mask req. Nov 12 - Jan 7': 0.897}
    # mask_scens = {'no mask req.': 1.0, 'mask req. on Nov. 12': 0.897}
    # mask_scens = {'no mask req.': 1.0}
    mask_scens = {'increased holiday transmission': 1.0, 'no mask req.': 1.0, 'short-term measures to reduce transmission through Jan 2': 0.89}
    mab_scens = {'15% uptake': 0.15 * 0.53, '50% uptake': 0.50 * 0.53}
    # mab_scens = {'50% uptake': 0.50 * 0.53}
    # mab_scens = {'15% uptake': 0.15 * 0.53}

    # mab_scens = {'15% uptake': 0.15 * 0.53, '30% uptake': 0.30 * 0.53, '50% uptake': 0.50 * 0.53}
    # mab_scens = {'15% uptake': 0.15 * 0.53}
    vacc_scens = json.load(open('input/vacc_proj_params.json'))
    # vacc_scens = {k: v for k, v in vacc_scens.items() if k != 'current trajectory w/o 5-11 vacc.'}
    # vacc_scens = {k: v for k, v in vacc_scens.items() if k == 'current trajectory'}

    scens_to_run = [
        # ('no mask req.', '15% uptake', 'current trajectory'),
        ('increased holiday transmission', '15% uptake', 'current trajectory'),
        # ('increased holiday transmission', '15% uptake', 'increased 5-11 vacc'),
        # ('increased holiday transmission', '15% uptake', 'increased booster uptake'),
        # ('increased holiday transmission', '15% uptake', '75% elig. boosters by Dec 31'),
        # ('increased holiday transmission', '15% uptake', 'increased 65+ boosters'),
        # ('increased holiday transmission', '15% uptake', 'increased under-65 boosters'),
        # ('increased holiday transmission', '15% uptake', 'increased adult vacc'),
        ('increased holiday transmission', '50% uptake', 'current trajectory'),
        ('increased holiday transmission', '15% uptake', 'max boosters and pede. vacc'),
        ('increased holiday transmission', '50% uptake', 'max boosters and pede. vacc'),
        ('short-term measures to reduce transmission through Jan 2', '15% uptake', 'current trajectory'),
        ('short-term measures to reduce transmission through Jan 2', '15% uptake', 'max boosters and pede. vacc'),
        ('short-term measures to reduce transmission through Jan 2', '50% uptake', 'max boosters and pede. vacc'),
        # ('mask req. on Nov. 12', '15% uptake', 'current trajectory'),
        # ('mask req. on Nov. 12', '50% uptake', '75% elig. boosters by Dec 31'),
        # ('mask req. Nov 12 - Jan 7', '15% uptake', 'current trajectory'),
        # ('mask req. Nov 12 - Jan 7', '50% uptake', '75% elig. boosters by Dec 31'),
        # ('proof of vacc. req. on Nov. 12', '15% uptake', 'current trajectory')
    ]

    base_mab_prevalence = pd.read_csv('input/mab_prevalence.csv', parse_dates=['date'], index_col=0)
    mab_prevalence_target_date = '2021-11-30'

    total_hosps_dict = {}
    peak_date_dict = {}
    peak_hosps_dict = {}
    future_deaths_dict = {}
    mean_census_hosps_dict = {}
    median_census_hosps_dict = {}
    median_new_infections_dict = {}
    scen_i = 0

    figs = {}
    axs = {}
    for s in ['base', 'vacc1', 'vacc2', 'vacc3', 'masks1', 'masks2', 'mab1', 'mab2', 'vaccpass1', 'vaccpass2', 'combined1', 'combined2', 'combined3', 'combined4']:
        figs[s], axs[s] = plt.subplots(figsize=(9, 6))
    colors = ['royalblue', 'firebrick', 'darkgreen', 'black']
    color_i = 0
    for mask_scen, mask_betta_mult in mask_scens.items():
        raw_params = base_raw_params.copy()
        b = raw_params['betta']
        raw_params['betta'] = {'tslices': [664, 668, 675, 698, 709], 'value': [b, b*mask_betta_mult, 1.18*b*mask_betta_mult, b*mask_betta_mult, 1.18*b*mask_betta_mult, b]}
        # raw_params['unvacc_relative_transm'] = {'tslices': [658], 'value': {'0-19': [1, 1 - 0.1*(1 - mask_betta_mult)], '20-39': [1, mask_betta_mult], '40-64': [1, mask_betta_mult], '65+': [1, mask_betta_mult]}}
        # raw_params['betta'] = {'tslices': [658, 714], 'value': [raw_params['betta'], raw_params['betta'] * mask_betta_mult, raw_params['betta']]}
        # raw_params['betta'] = {'tslices': [658], 'value': [raw_params['betta'], raw_params['betta'] * mask_betta_mult]}
        for mab_scen, mab_prevalence_target in mab_scens.items():
            mab_prevalence = build_mab_prevalence(base_mab_prevalence, mab_prevalence_target, mab_prevalence_target_date)
            for vacc_scen, vacc_proj_params in vacc_scens.items():
                if (mask_scen, mab_scen, vacc_scen) in scens_to_run:
                # if True:
                    print((mask_scen, mab_scen, vacc_scen))
                    scen_i += 1
                    scen_index = (mask_scen, mab_scen, vacc_scen)
                    dirname = f'{data_dir}/{", ".join(scen_index)}'
                    Path(dirname).mkdir(parents=True, exist_ok=True)
                    if get_from_files:
                        tc_sims, census_hosps, new_infections, new_hosps, total_deaths = get_sims_from_files(dirname)
                    else:
                        print(f'Running simulations for scenario #{scen_i}: {scen_index}')
                        tc_sims, census_hosps, new_infections, new_hosps, total_deaths = get_sims(fit, engine, max_date=dt.datetime(2022, 5, 31),
                                                                                 sample_count=1000,
                                                                                 model_params={'params': raw_params, 'vacc_proj_params': vacc_proj_params, 'mab_prevalence': mab_prevalence},
                                                                                 output_dir=dirname)

                    peak_hosps_dict[scen_index] = census_hosps.loc[start_date:'2021-12-31'].max(axis=0)
                    peak_date_dict[scen_index] = census_hosps.loc[start_date:'2021-12-31'].idxmax(axis=0)
                    total_hosps_dict[scen_index] = new_hosps.loc[start_date:'2022-05-30'].sum(axis=0)
                    future_deaths_dict[scen_index] = total_deaths.loc['2022-05-30'] - total_deaths.loc[start_date]

                    median_census_hosps_dict[scen_index] = census_hosps.loc[start_date:'2021-12-31'].median(axis=1)
                    mean_census_hosps_dict[scen_index] = census_hosps.loc[start_date:'2021-12-31'].mean(axis=1)
                    median_new_infections_dict[scen_index] = new_infections.median(axis=1)

                    # base
                    if scen_index == ('increased holiday transmission', '15% uptake', 'current trajectory'):
                        # census_hosps.mean(axis=1).plot(ax=axs['base'], color='navy', linestyle='dashed')
                        census_hosps.median(axis=1).plot(ax=axs['base'], color='navy', label='Median')
                        plot_prediction_interval(census_hosps, ax=axs['base'], color='navy', label='50% Prediction Interval', ci=0.50, alpha=0.6)
                        plot_prediction_interval(census_hosps, ax=axs['base'], color='navy', label='90% Prediction Interval', ci=0.90, alpha=0.3)

                        for s, ax in axs.items():
                            if 'vacc' in s:
                                plot_prediction_interval(census_hosps, ax=ax, color=colors[0], label=vacc_scen, ci=0.50)
                                # census_hosps.median(axis=1).plot(ax=ax, color=colors[0], label='_nolegend_')
                            if 'mask' in s:
                                plot_prediction_interval(census_hosps, ax=ax, color=colors[0], label=mask_scen, ci=0.50)
                            if 'mab' in s:
                                plot_prediction_interval(census_hosps, ax=ax, color=colors[0], label='current uptake', ci=0.50)
                            if 'vaccpass' in s:
                                plot_prediction_interval(census_hosps, ax=ax, color=colors[0], label=mask_scen, ci=0.50)
                            if 'combined' in s:
                                census_hosps.median(axis=1).plot(ax=ax, color=colors[0], label='current trajectory')
                                # plot_prediction_interval(census_hosps, ax=ax, color=colors[0], label='current trajectory', ci=0.50)

                    # boosters
                    # if scen_index == ('no mask req.', '15% uptake', 'increased booster uptake'):
                    if scen_index == ('increased holiday transmission', '15% uptake', 'max boosters and pede. vacc'):
                        plot_prediction_interval(census_hosps, ax=axs['vacc2'], color=colors[1], label='large increase in boosters and pede. vacc.', ci=0.50)
                        plot_prediction_interval(census_hosps, ax=axs['vacc3'], color=colors[1], label='large increase in boosters and pede. vacc.', ci=0.50)
                        # census_hosps.median(axis=1).plot(ax=axs['vacc2'], color=colors[1], label='_nolegend_')
                    if scen_index == ('increased holiday transmission', '15% uptake', '75% elig. boosters by Dec 31'):
                        plot_prediction_interval(census_hosps, ax=axs['vacc3'], color=colors[2], label=vacc_scen, ci=0.50)

                    # mab
                    if scen_index == ('increased holiday transmission', '50% uptake', 'current trajectory'):
                        plot_prediction_interval(census_hosps, ax=axs['mab2'], color=colors[1], label='50% mAb uptake', ci=0.50)

                    # masks
                    if scen_index == ('short-term measures to reduce transmission through Jan 2', '15% uptake', 'current trajectory'):
                        plot_prediction_interval(census_hosps, ax=axs['masks2'], color=colors[1], label=mask_scen, ci=0.50)

                    # vaccpass
                    if scen_index == ('proof of vacc. req. on Nov. 12', '15% uptake', 'current trajectory'):
                        plot_prediction_interval(census_hosps, ax=axs['vaccpass2'], color=colors[1], label=mask_scen, ci=0.50)

                    # # combined
                    # if scen_index == ('increased holiday transmission', '15% uptake', 'max boosters and pede. vacc'):
                    #     plot_prediction_interval(census_hosps, ax=axs['combined2'], color=colors[1], label='large increase in boosters and pede. vacc.', ci=0.50)
                    #     plot_prediction_interval(census_hosps, ax=axs['combined3'], color=colors[1], label='large increase in boosters and pede. vacc.', ci=0.50)
                    #     plot_prediction_interval(census_hosps, ax=axs['combined4'], color=colors[1], label='large increase in boosters and pede. vacc.', ci=0.50)
                    #
                    # if scen_index == ('short-term measures to reduce transmission through Jan 2', '15% uptake', 'max boosters and pede. vacc'):
                    #         plot_prediction_interval(census_hosps, ax=axs['combined3'], color=colors[2], label='...and increased mAb uptake', ci=0.50)
                    #         # plot_prediction_interval(census_hosps, ax=axs['combined4'], color=colors[2],label='...and increased mAb uptake', ci=0.50)
                    #
                    # if scen_index == ('increased holiday transmission', '50% uptake', 'max boosters and pede. vacc'):
                    #         plot_prediction_interval(census_hosps, ax=axs['combined4'], color=colors[2], label='...and short-term measures to reduce transmission through Jan 2', ci=0.50)

                   # combined
                    if scen_index == ('increased holiday transmission', '15% uptake', 'max boosters and pede. vacc'):
                        census_hosps.median(axis=1).plot(ax=axs['combined2'], color=colors[1], label='...with a large increase in booster and pede. vacc.')
                        census_hosps.median(axis=1).plot(ax=axs['combined3'], color=colors[1], label='...with a large increase in booster and pede. vacc.')
                        census_hosps.median(axis=1).plot(ax=axs['combined4'], color=colors[1], label='...with a large increase in booster and pede. vacc.')

                    if scen_index == ('increased holiday transmission', '50% uptake', 'max boosters and pede. vacc'):
                            census_hosps.median(axis=1).plot(ax=axs['combined3'], color=colors[2], label='...and increased mAb uptake')
                            census_hosps.median(axis=1).plot(ax=axs['combined4'], color=colors[2],label='...and increased mAb uptake')

                    if scen_index == ('short-term measures to reduce transmission through Jan 2', '50% uptake', 'max boosters and pede. vacc'):
                            census_hosps.median(axis=1).plot(ax=axs['combined4'], color=colors[3],label='...and a temporary small reduction in transmission through Jan 3')

    # exit()

    for s, ax in axs.items():
        ax.legend(loc='upper left')
        ax.set_xlim(dt.datetime(2020, 3, 1), dt.datetime(2022, 5, 31))
        ax.set_ylim(0, 2500)
        ax.set_ylabel('Hospitalized with COVID-19')

    for s, fig in figs.items():
        print(s)
        fig.savefig(f'{data_dir}/{s}.png', bbox_inches='tight')

    # exit()


    end_date_str = census_hosps.index.max().strftime("%b %#d, %Y")

    scen_type_names = ['Mask Order Scenario', 'mAb Scenario', 'Vaccine Scenario']
    index_names = scen_type_names + ['Date']

    total_hosps = pd.concat(total_hosps_dict).rename_axis(index=index_names)
    peak_hosps = pd.concat(peak_hosps_dict).rename_axis(index=index_names)
    median_census_hosps = pd.concat(median_census_hosps_dict).rename_axis(index=index_names)
    mean_census_hosps = pd.concat(mean_census_hosps_dict).rename_axis(index=index_names)
    median_new_infections = pd.concat(median_new_infections_dict).rename_axis(index=index_names)
    future_deaths = pd.concat(future_deaths_dict).rename_axis(index=index_names)
    peak_dates = pd.concat(peak_date_dict).rename_axis(index=index_names)

    # median_census_hosps.reset_index(['Vaccine Passport Scenario', 'Mask Order Scenario', 'Vaccine Scenario']).unstack(level='mAb Scenario').plot()
    # actual_hosps(engine)
    # plt.legend(loc='upper left')
    # plt.show()
    # exit()

    results = pd.DataFrame(index=total_hosps.index.droplevel('Date').unique())
    results[f'Total Hospitalizations Between Now and {end_date_str}'] = total_hosps.groupby(scen_type_names).mean().div(10).astype(int).mul(10)
    results[f'Total Deaths Between Now and {end_date_str}'] = future_deaths.groupby(scen_type_names).mean().div(10).astype(int).mul(10)
    results[f'Median of Peak Hospitalizaions'] = peak_hosps.groupby(scen_type_names).median()
    results[f'Mean of Peak Hospitalizaions'] = peak_hosps.groupby(scen_type_names).mean()
    results[f'Peak of Median Hospitalizations'] = median_census_hosps.groupby(scen_type_names).max()
    results[f'Peak of Mean Hospitalizations'] = mean_census_hosps.groupby(scen_type_names).max()
    results[f'Median Date of Peak'] = peak_dates.groupby(scen_type_names).median()
    # results[f'Median Total Hospitalizations Between Now and {end_date_str}'] = total_hosps.groupby(scen_type_names).median().astype(int)
    for hosp_peak_limit in [2000]:
        results[f'Probability of Exceeding {hosp_peak_limit} Hospitalized Before Dec 31, 2021'] = (1*(peak_hosps > hosp_peak_limit)).groupby(scen_type_names).mean()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('display.width', 1000)
    print(results.sort_index())

    results.sort_index().transpose().to_csv(f'{data_dir}/summary.csv')

    # data = pd.concat(data)
    # plot_kde(new_hosp_sims.loc[start_date:].sum(axis=0), ax=ax, ci=None,
    #          xlabel=f'Total Hospital Admissions From Now Through {end_date_str}', label=scen_label, color=color)
