import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import json
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates, format_date_axis, modeled_re
from time import perf_counter
from timeit import timeit

from covid_model.db import db_engine
from covid_model.model_with_omicron import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications


def omicron_prevalence_share(model):
    df = model.solution_sum(['seir', 'variant'])
    prevalence_by_variant = df.xs('I', axis=1, level='seir') + df.xs('A', axis=1, level='seir')
    return prevalence_by_variant['omicron'] / prevalence_by_variant.sum(axis=1)


def omicron_re(model):
    df = model.solution_sum(['seir', 'variant']).xs('omicron', axis=1, level='variant')
    new_infections = df['E'] / model.specifications.model_params['alpha']
    infect_duration = 1 / model.specifications.model_params['gamm']
    infected = (df['I'].shift(3) + df['A'].shift(3))
    return infect_duration * new_infections / infected


def plot(model, ax, label=None, min_date=dt.date(2021, 7, 1), max_date=dt.date(2022, 3, 31), max_hosps=4000):
    modeled_hosps = model.solution_sum('seir')['Ih']

    if max_hosps and modeled_hosps.max() > max_hosps:
        index_where_passes_max = modeled_hosps[modeled_hosps > max_hosps].index.tolist()[0]
        modeled_hosps = modeled_hosps.loc[:index_where_passes_max]
    ax.plot(model.daterange[:len(modeled_hosps)], modeled_hosps, label=label)
    ax.set_xlim((min_date, max_date))
    ax.set_ylim((0, max_hosps))

    format_date_axis(ax)
    ax.grid(color='lightgray')


if __name__ == '__main__':
    engine = db_engine()

    model = CovidModelWithVariants(end_date=dt.date(2022, 12, 31))
    # cms = CovidModelSpecifications.from_db(engine, 414, new_end_date=model.end_date)
    # cms.set_model_params('input/params.json')

    fig, axs = plt.subplots(2, 2)
    print('Prepping model...')
    print(timeit("model.prep(specs=417, engine=engine)", number=1, globals=globals()), 'seconds to prep model.')
    print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')
    for ax in axs.flatten():
        plot(model, ax=ax, label='No Omicron')
        # modeled(model, 'Ih', ax=ax, label='No Omicron')
        # ax.plot(model.daterange, omicron_re(model), label='No Omicron')
        # plt.plot(model.daterange, omicron_prevalence_share(model), label='No Omicron')

    scenarios = {
        # 'Low Infectiousness, Low Immune-Escape': {'omicron_transm_mult': 1.5*5/3, 'omicron_lp': 1.5, 'omicron_ip': 3, 'omicron_acq_immune_escape': 0.33, 'omicron_vacc_eff_k': 2.65},
        # 'Low Infectiousness, High Immune-Escape': {'omicron_transm_mult': 1.5*5/3, 'omicron_lp': 1.5, 'omicron_ip': 3, 'omicron_acq_immune_escape': 0.89, 'omicron_vacc_eff_k': 7},
        # 'High Infectiousness, Low Immune-Escape': {'omicron_transm_mult': 2.5*5/3, 'omicron_lp': 1.5, 'omicron_ip': 3, 'omicron_acq_immune_escape': 0.33, 'omicron_vacc_eff_k': 2.65},
        'High Infectiousness, High Immune-Escape': {'omicron_transm_mult': 2.5*5/3, 'omicron_lp': 1.5, 'omicron_ip': 3, 'omicron_acq_immune_escape': 0.89, 'omicron_vacc_eff_k': 7}
    }

    virulence = {
        # '30% Reduced Virulence': 0.70,
        # '60% Reduced Virulence': 0.40,
        '90% Reduced Virulence': 0.10,
        # '97% Reduced Virulence': 0.03,
    }

    for om_imports_start in [0.5]:
        om_imports_total = 1000
        om_imports_by_day = om_imports_start * np.power(2, np.arange(999) / 14)
        om_imports_by_day = om_imports_by_day[om_imports_by_day.cumsum() <= om_imports_total]
        om_imports_by_day[-1] += om_imports_total - om_imports_by_day.sum()
        print(f'Omicron imports by day (over {len(om_imports_by_day)} days): {om_imports_by_day}')
        for t, n in zip(668 + np.arange(len(om_imports_by_day)), om_imports_by_day):
            model.set_param('om_seed', n, trange=[t])
        base_params = model.params.copy()
        for ax, (scen_label, scen_params) in zip(axs.flatten(), scenarios.items()):
            ax.title.set_text(scen_label)
            for param, val in scen_params.items():
                if param == 'omicron_vacc_eff_k':
                    model.specifications.model_params[param] = val
                    print(timeit('model.apply_omicron_vacc_eff()', number=1, globals=globals()), f'seconds to reapply specifications with {param} = {val}.')
                elif param == 'omicron_lp':
                    model.set_param('alpha', val, attrs={'variant': 'omicron'})
                elif param == 'omicron_ip':
                    model.set_param('gamm', 1/val, attrs={'variant': 'omicron'})
                else:
                    model.set_param(param, val)

            for virulence_label, hosp_mult in virulence.items():
                model.set_param('hosp', mult=hosp_mult, attrs={'variant': 'omicron'})
                model.set_param('dnh', mult=hosp_mult, attrs={'variant': 'omicron'})
                print(model.params[680])
                model.build_ode()
                print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')
                plot(model, ax=ax, label=virulence_label)
                # modeled(model, 'Ih', ax=ax, label=virulence_label)
                # ax.plot(model.daterange, omicron_re(model), label=virulence_label)
                model.set_param('hosp', mult=1/hosp_mult, attrs={'variant': 'omicron'})
                model.set_param('dnh', mult=1/hosp_mult, attrs={'variant': 'omicron'})
            ax.legend(loc='upper left')

            # model.build_ode()
            # print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')
            # modeled(model, 'Ih', ax=ax, label=scen_label)
            # plt.plot(model.daterange, omicron_prevalence_share(model), label=scen_label)
            # modeled_re(model, ax=ax, label=scen_label)


        # for omicron_transm_mult in [1, 1.5, 2]:
        # for omicron_transm_mult in [1, 1.5, 2, 3]:
        #     model.set_param('omicron_transm_mult', omicron_transm_mult)
        #     # for omicron_immune_escape in [0, 0.2, 0.5, 0.8]:
        #     for omicron_immune_escape in [0, 0.5, 0.8, 1]:
        #         model.set_param('omicron_acq_immune_escape', omicron_immune_escape)
        #         model.set_param('vacc_eff', mult=(1 - omicron_immune_escape), attrs={'vacc': 'vacc', 'variant': 'omicron'})
        #         model.build_ode()
        #         print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')
        #         # modeled(model, 'Ih', ax=ax, label=f'Import {int(sum(om_imports_by_week))} Omicron cases over the next {len(om_imports_by_week) - 1} weeks; {omicron_transm_mult}x as infectious as Delta; {100*omicron_immune_escape}% immune-escape')
        #         modeled_re(model, ax=ax, label=f'Import {int(sum(om_imports_by_day))} Omicron cases over {len(om_imports_by_day) - 1} days; {omicron_transm_mult}x as infectious as Delta; {100 * omicron_immune_escape}% immune-escape')

    # model.write_to_db(engine)
    # actual_hosps(engine)
    # ax.legend(loc='upper left')
    for ax in axs.flatten():
        format_date_axis(ax)
        # ax.set_xlim((dt.date(2021, 7, 1), dt.date(2023, 1, 1)))
        # ax.set_ylim((0, 5))

    fig.tight_layout()
    plt.show()
