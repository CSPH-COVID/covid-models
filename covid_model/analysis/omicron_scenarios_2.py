import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import datetime as dt
import copy
import json
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates, format_date_axis, modeled_re
from time import perf_counter
from timeit import timeit
import re

from covid_model.db import db_engine
from covid_model.model_with_omicron import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications


def plot(model, ax, method=None, compartments=['Ih'], label=None, min_date=dt.date(2021, 7, 1), max_date=dt.date(2022, 2, 28),
         clip_after_max=False, max_value=None, per_population=None):
    if method is not None:
        modeled_values = method(model)
    else:
        modeled_values = model.solution_sum('seir')[compartments].sum(axis=1)

    if per_population:
        modeled_values *= per_population / model.specifications.model_params['total_pop']

    if clip_after_max and max_value and modeled_values.max() > max_value:
        index_where_passes_max = modeled_values[modeled_values > max_value].index.tolist()[0]
        modeled_values = modeled_values.loc[:index_where_passes_max]
    ax.plot(model.daterange[:len(modeled_values)], modeled_values, label=label)
    ax.set_xlim((min_date, max_date))
    ax.set_ylim((0, max_value))

    format_date_axis(ax, interval_months=1)
    ax.grid(color='lightgray')


def omicron_prevalence_share(model):
    df = model.solution_sum(['seir', 'variant'])
    prevalence_by_variant = df.xs('I', axis=1, level='seir') + df.xs('A', axis=1, level='seir')
    return prevalence_by_variant['omicron'] / prevalence_by_variant.sum(axis=1)


def apply_omicron_params(model, omicron_params, base_params=None):
    if base_params is not None:
        model.params = copy.deepcopy(base_params)

    for param, val in omicron_params.items():
        if param == 'omicron_vacc_eff_k':
            model.specifications.model_params[param] = val
            print(timeit('model.apply_omicron_vacc_eff()', number=1, globals=globals()),
                  f'seconds to reapply specifications with {param} = {val}.')
        elif param == 'seed_t':
            apply_omicron_seed(model, start_t=val, start_rate=0.25, total=100)
        elif param[:14] == 'vacc_succ_hosp':
            model.set_param('hosp', val, attrs={'variant': 'omicron', 'vacc': 'vacc', 'age': param[15:]})
        elif re.match(r'\w+_mult_\d{8}_\d{8}', param):
            str_parts = param.split('_')
            from_date = dt.datetime.strptime(str_parts[-2], '%Y%m%d').date()
            to_date = dt.datetime.strptime(str_parts[-1], '%Y%m%d').date()
            param = param[:-23]
            model.set_param(param, mult=val, trange=range((from_date - model.start_date).days, (to_date - model.start_date).days))
        # elif param == 'betta_mult_20211221':
        #     start_t = (dt.date(2021, 12, 17) - model.start_date).days
        #     model.set_param('betta', mult=val, trange=range(start_t, start_t + 16))
        #     # model.set_param('betta', mult=1 - 0.67 * (1 - val), trange=range(start_t + 16, start_t + 42))
        #     model.set_param('betta', mult=1 - 1 * (1 - val), trange=range(start_t + 16, start_t + 42))
        elif param[-5:] == '_mult':
            model.set_param(param[:-5], mult=val, attrs={'variant': 'omicron'})
        else:
            model.set_param(param, val, attrs={'variant': 'omicron'})

    model.build_ode()


def apply_omicron_seed(model, start_t=668, start_rate=0.5, total=1000, doubling_time=14):
    om_imports_by_day = start_rate * np.power(2, np.arange(999) / doubling_time)
    om_imports_by_day = om_imports_by_day[om_imports_by_day.cumsum() <= total]
    om_imports_by_day[-1] += total - om_imports_by_day.sum()
    print(f'Omicron imports by day (over {len(om_imports_by_day)} days): {[round(x) for x in om_imports_by_day]}')
    for t, n in zip(start_t + np.arange(len(om_imports_by_day)), om_imports_by_day):
        model.set_param('om_seed', n, trange=[t])

    model.build_ode()


generic_omicron_params = {'alpha': 1.5, 'gamm': 1/3, 'hlos_mult': 0.60, 'omicron_acq_immune_escape': 0.89, 'omicron_vacc_eff_k': 7}

scenarios = {
        # '5-Week Wave: More infectious; 95% reduced hosp. rate':
        #     {'seed_t': 666, 'betta_mult': 2.03*5/3, 'hosp_mult': 0.05, 'dnh_mult': 0.05},
        '6-Week Wave: Less infectious; 90% reduced hosp. rate':
            {'seed_t': 659, 'betta_mult': 1.65*5/3, 'hosp_mult': 0.075, 'dnh_mult': 0.075},
        # '8-Week Wave: More infectious; flattened by increase in TC Dec 17 - Jan 28; 85% reduced hosp. rate':
        #     {'seed_t': 664, 'betta_mult': 2*5/3, 'hosp_mult': 0.15, 'dnh_mult': 0.15, 'betta_mult_20211221': 0.7},
        # '10-Week Wave: Less infectious; flattened by increase in TC Dec 17 - Jan 28; 80% reduced hosp. rate':
        #     {'seed_t': 660, 'betta_mult': 1.73*5/3, 'hosp_mult': 0.192, 'dnh_mult': 0.2, 'betta_mult_20211221': 0.7, 'hlos_mult': 1.0},

        'New Scenario 1':
            {'seed_t': 664, 'betta_mult': 2.4*5/3, 'hosp_mult': 0.245, 'dnh_mult': 0.20, 'omicron_acq_immune_escape': 0.56, 'omicron_vacc_eff_k': 4.5,
             'betta_mult_20211206_20211220': 0.85, 'betta_mult_20211220_20220103': 0.6, 'betta_mult_20220103_20220128': 0.55},
        # 'New Scenario 2':
        #     {'seed_t': 659, 'betta_mult': 2.5*5/3, 'hosp_mult': 0.21, 'dnh_mult': 0.20, 'omicron_acq_immune_escape': 0.3, 'omicron_vacc_eff_k': 3,
        #      'betta_mult_20211213_20220128': 0.7},

        # 'New Scenario 3':
        #     {'seed_t': 659, 'betta_mult': 1.65*5/3, 'hosp_mult': 0.075, 'dnh_mult': 0.075, 'vacc_succ_hosp_65+': 0.001, 'vacc_succ_hosp_40-64': 0.0005},

    # 'New Scenario':
        #     {'seed_t': 650, 'betta_mult': 1.6*5/3, 'hosp_mult': 0.10, 'dnh_mult': 0.075, 'omicron_acq_immune_escape': 0.56, 'omicron_vacc_eff_k': 4.5},
        # 'New Scenario 2':
        #     {'seed_t': 655, 'betta_mult': 1.85*5/3, 'hosp_mult': 0.175, 'dnh_mult': 0.075, 'omicron_acq_immune_escape': 0.56, 'omicron_vacc_eff_k': 4.5, 'betta_mult_20211221': 0.7}
   }

# scenarios = {
#     '10-Week Wave, no NPIs':
#             {'seed_t': 660, 'betta_mult': 1.73*5/3, 'hosp_mult': 0.192, 'dnh_mult': 0.2, 'betta_mult_20211221': 0.7, 'hlos_mult': 1.0},
#     '10-Week Wave, 12-day NPI starting Jan 7 (40% reduction in transm.)':
#             {'seed_t': 660, 'betta_mult': 1.73*5/3, 'hosp_mult': 0.192, 'dnh_mult': 0.2, 'betta_mult_20211221': 0.7, 'hlos_mult': 1.0,
#              'betta_mult_20220107_20220119': 0.60},
#     '10-Week Wave, 12-day NPI starting Jan 10 (40% reduction in transm.)':
#             {'seed_t': 660, 'betta_mult': 1.73*5/3, 'hosp_mult': 0.192, 'dnh_mult': 0.2, 'betta_mult_20211221': 0.7, 'hlos_mult': 1.0,
#              'betta_mult_20220110_20220122': 0.60},
#     '10-Week Wave, 12-day NPI starting Jan 13 (40% reduction in transm.)':
#             {'seed_t': 660, 'betta_mult': 1.73*5/3, 'hosp_mult': 0.192, 'dnh_mult': 0.2, 'betta_mult_20211221': 0.7, 'hlos_mult': 1.0,
#              'betta_mult_20220113_20220125': 0.60}
#    }


if __name__ == '__main__':
    engine = db_engine()

    model = CovidModelWithVariants(end_date=dt.date(2022, 3, 1))

    figs = {}
    axs = {}
    for title in [
        'Hospitalized with Covid-19',
        # 'Hospitalized with Covid-19 (with two-week NPI)',
        'Infected with SARS-CoV-2',
        'Share of Current Infections That Are Omicron']:
        fig, ax = plt.subplots()
        figs[title] = fig
        axs[title] = ax
        ax.set_ylabel(title)
        fig.tight_layout()

    print('Prepping model...')
    model.prep(specs=521, engine=engine, mab_prevalence='input/mab_prevalence.csv', param_multipliers='input/param_multipliers.json')
    model.solve_seir()
    plot(model, ax=axs['Hospitalized with Covid-19'], compartments=['Ih'], label='No Omicron')
    # plot(model, ax=axs['Hospitalized with Covid-19 (with two-week NPI)'], compartments=['Ih'], label='No Omicron')
    plot(model, ax=axs['Infected with SARS-CoV-2'], compartments=['I', 'A'], label='No Omicron', per_population=True)
    plot(model, ax=axs['Share of Current Infections That Are Omicron'], method=omicron_prevalence_share, label='No Omicron')

    # apply_omicron_seed(model, start_t=668, total=100)
    base_params = copy.deepcopy(model.params)

    hosps_df = pd.DataFrame(index=model.trange)
    hosps_df['Without Omicron'] = model.solution_sum('seir')[['Ih']].sum(axis=1)
    preval_df = pd.DataFrame(index=model.trange)
    preval_df['Without Omicron'] = model.solution_sum('seir')[['I', 'A']].sum(axis=1) / model.specifications.model_params['total_pop']
    for scen_label, scen_params in scenarios.items():
        print(f'Building scenario "{scen_label}"...')
        apply_omicron_params(model, omicron_params={**generic_omicron_params, **scen_params}, base_params=base_params)
        model.solve_seir()
        plot(model, ax=axs['Hospitalized with Covid-19'], compartments=['Ih'], label=scen_label, max_value=4000)
        plot(model, ax=axs['Infected with SARS-CoV-2'], compartments=['I', 'A'], label=scen_label, per_population=True, min_date=dt.date(2020, 3, 1))
        plot(model, ax=axs['Share of Current Infections That Are Omicron'], method=omicron_prevalence_share, label=scen_label)

        hosps_df[scen_label] = model.solution_sum('seir')[['Ih']].sum(axis=1)
        preval_df[scen_label] = model.solution_sum('seir')[['I', 'A']].sum(axis=1) / model.specifications.model_params['total_pop']

    hosps_df.to_csv('output/omicron_scenarios_hosps.csv')
    preval_df.to_csv('output/omicron_scenarios_preval.csv')

    axs['Infected with SARS-CoV-2'].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    actual_hosps(engine, ax=axs['Hospitalized with Covid-19'], color='black', label='EMResource Hospitalizations')
    for ax in axs.values():
        ax.legend(loc='upper left')
        ax.autoscale(True, axis='y')
        ax.grid(color='lightgray')
        ax.set_xlim(dt.date(2021, 7, 1), dt.date(2022, 2, 28))
        # ax.set_xlim(dt.date(2022, 1, 1), dt.date(2022, 1, 20))

    plt.show()
