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


if __name__ == '__main__':
    engine = db_engine()

    model = CovidModelWithVariants(end_date=dt.date(2022, 12, 31))
    cms = CovidModelSpecifications.from_db(engine, 386, new_end_date=model.end_date)
    cms.set_model_params('input/params.json')

    model.prep(specs=cms)

    fig, axs = plt.subplots(2, 3)
    axs = axs.transpose()
    print('Prepping model...')
    print(timeit('model.prep()', number=1, globals=globals()), 'seconds to prep model.')
    print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')
    for ax in axs.flatten():
        modeled(model, 'Ih', ax=ax, label='No Omicron')
        # ax.plot(model.daterange, omicron_re(model), label='No Omicron')
        # plt.plot(model.daterange, omicron_prevalence_share(model), label='No Omicron')

    scenarios = {
        # 'Low Vacc Escape': {'omicron_transm_mult': 2, 'omicron_acq_immune_escape': 0.5, 'omicron_vacc_eff_k': 1.6},
        # 'High Vacc Escape': {'omicron_transm_mult': 2, 'omicron_acq_immune_escape': 0.5, 'omicron_vacc_eff_k': 2.6},
        'Low Infectiousness, High Immune-Escape (vs. Acq. Immunity Only)': {'omicron_transm_mult': 1, 'omicron_acq_immune_escape': 0.89, 'omicron_vacc_immune_escape': 0},
        'Low Infectiousness, High Immune-Escape (Including vs. Vacc.)': {'omicron_transm_mult': 1, 'omicron_acq_immune_escape': 0.89, 'omicron_vacc_immune_escape': 0.89},
        'Med. Infectiousness, Med. Immune-Escape (vs. Acq. Immunity Only)': {'omicron_transm_mult': 1.5, 'omicron_acq_immune_escape': 0.55, 'omicron_vacc_immune_escape': 0},
        'Med. Infectiousness, Med. Immune-Escape (Including vs. Vacc.)': {'omicron_transm_mult': 1.5, 'omicron_acq_immune_escape': 0.55, 'omicron_vacc_immune_escape': 0.55},
        'High Infectiousness, Low Immune-Escape (vs. Acq. Immunity Only)': {'omicron_transm_mult': 2, 'omicron_acq_immune_escape': 0.22, 'omicron_vacc_immune_escape': 0},
        'High Infectiousness, Low Immune-Escape (Including vs. Vacc.)': {'omicron_transm_mult': 2, 'omicron_acq_immune_escape': 0.22, 'omicron_vacc_immune_escape': 0.22},
    }

    virulence = {
        'Same Virulence': 1.0,
        # '25% Reduced Virulence': 0.75,
        # '50% Reduced Virulence': 0.5
    }

    for om_imports_start in [1]:
        om_imports_total = 2000
        om_imports_by_day = np.round(om_imports_start * np.power(2, np.arange(999) / 14))
        om_imports_by_day = om_imports_by_day[om_imports_by_day.cumsum() <= om_imports_total]
        om_imports_by_day[-1] += om_imports_total - om_imports_by_day.sum()
        print(f'Omicron imports by day (over {len(om_imports_by_day)} days): {om_imports_by_day}')
        for t, n in zip(668 + np.arange(len(om_imports_by_day)), om_imports_by_day):
            model.set_param('om_seed', n, trange=[t])
        for ax, (scen_label, scen_params) in zip(axs.flatten(), scenarios.items()):
            ax.title.set_text(scen_label)
            for param, val in scen_params.items():
                if param == 'omicron_vacc_eff_k':
                    model.specifications.model_params[param] = val
                    print(timeit('model.apply_omicron_vacc_eff()', number=1, globals=globals()), f'seconds to reapply specifications with {param} = {val}.')
                else:
                    model.set_param(param, val)
            for virulence_label, hosp_mult in virulence.items():
                model.set_param('hosp', mult=hosp_mult, attrs={'variant': 'omicron'})
                model.set_param('dnh', mult=hosp_mult, attrs={'variant': 'omicron'})
                model.build_ode()
                print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')
                modeled(model, 'Ih', ax=ax, label=virulence_label)
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
        # ax.set_xlim((dt.date(2021, 1, 1), dt.date(2023, 1, 1)))
        # ax.set_ylim((0, 5))

    fig.tight_layout()
    plt.show()
