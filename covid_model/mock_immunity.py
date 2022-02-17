import datetime as dt
from matplotlib import pyplot as plt, ticker as mtick
from cycler import cycler
import argparse

from covid_model.db import db_engine
from covid_model.ode_builder import *
from model_with_immunity_rework import CovidModelWithVariants


def build_default_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--days", type=int, help="the number of days of immunity to plot")
    parser.set_defaults(days=180)
    clargs = parser.parse_args()

    model = CovidModelWithVariants(end_date=CovidModelWithVariants.default_start_date + dt.timedelta(clargs.days=10))
    model.set_specifications(702, engine=engine, params='input/params.json', attribute_multipliers='input/attribute_multipliers.json')
    model.apply_specifications(apply_vaccines=False)
    model.set_param('shot1_per_available', 0)
    model.set_param('shot2_per_available', 0)
    model.set_param('shot3_per_available', 0)
    model.set_param('betta', 0)

    return model


if __name__ == '__main__':
    engine = db_engine()

    variants = {
        'Non-Omicron': 'none',
        'Omicron': 'omicron'
    }

    immunities = {
        'Dose-2': {'initial_attrs': {'seir': 'S'}, 'params': {f'shot{i}_per_available': 1 for i in [1, 2]}},
        'Booster': {'initial_attrs': {'seir': 'S'}, 'params': {f'shot{i}_per_available': 1 for i in [1, 2, 3]}},
        'Prior Delta Infection': {'initial_attrs': {'seir': 'E', 'variant': 'none'}, 'params': {}},
        'Prior Omicron Infection': {'initial_attrs': {'seir': 'E', 'variant': 'omicron'}, 'params': {}}
    }

    fig, axs = plt.subplots(2, 2)

    for (immunity_label, immunity_specs), ax in zip(immunities.items(), axs.flatten()):
        ax.set_prop_cycle(cycler(color=['paleturquoise', 'darkcyan', 'violet', 'indigo']))
        ax.set_title(f'Immunity from {immunity_label}')

        print(f'Prepping and running model for {immunity_label} immunity...')
        model = build_default_model()
        for k, v in immunity_specs['params'].items():
            model.set_param(k, v)
        model.build_ode()
        model.compile()
        model.solve_ode({model.get_default_cmpt_by_attrs({**immunity_specs['initial_attrs'], 'age': age}): n for age, n in model.specifications.group_pops.items()})

        params = model.params_as_df
        group_by_attr_names = ['seir'] + [attr_name for attr_name in model.param_attr_names if attr_name != 'variant']
        n = model.solution_sum(group_by_attr_names).stack(level=group_by_attr_names).xs('S', level='seir')

        for variant_label, variant in variants.items():
            if immunity_label != 'Prior Omicron Infection' or variant_label == 'Non-Omicron':
                variant_params = params.xs(variant, level='variant')

                net_severe_immunity = (n * (1 - (1 - variant_params['immunity']) * (1 - variant_params['severe_immunity']))).groupby('t').sum() / n.groupby('t').sum()
                net_severe_immunity.plot(label=f'Immunity vs Severe {"Disease" if immunity_label == "Prior Omicron Infection" else variant_label}', ax=ax)

                immunity = (n * variant_params['immunity']).groupby('t').sum() / n.groupby('t').sum()
                immunity.plot(label=f'Immunity vs {"Infection" if immunity_label == "Prior Omicron Infection" else variant_label}', ax=ax)

        ax.legend(loc='best')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(color='lightgray')
        ax.set_xlabel(f'Days Since {immunity_label}')
        ax.set_ylim((0, 1))
        ax.set_xticks(np.arange(0, 365, 30))
        ax.set_xlim((30, 360))

    fig.tight_layout()
    plt.show()