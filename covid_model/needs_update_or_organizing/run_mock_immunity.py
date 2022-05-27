### Python Standard Library ###
import datetime as dt
import os
### Third Party Imports ###
from cycler import cycler
import numpy as np
from matplotlib import pyplot as plt, ticker as mtick
### Local Imports ###
from covid_model.utils import get_filepath_prefix
from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser


def build_default_model(days, engine, base_model=None, **specs_args):
    if base_model is None:
        base_model = CovidModel(engine=engine, **specs_args)
    model = CovidModel(base_model=base_model, end_date=base_model.start_date + dt.timedelta(days=days))
    model.build_param_lookups()
    model.set_param('shot1_per_available', 0)
    model.set_param('shot2_per_available', 0)
    model.set_param('shot3_per_available', 0)
    model.set_param('betta', 0)
    model.set_param('initial_seed', 0)
    model.set_param('alpha_seed', 0)
    model.set_param('delta_seed', 0)
    model.set_param('omicron_seed', 0)
    model.set_param('ba2_seed', 0)

    return model


def run_mock_immunity(days, outdir=None, base_model=None, fname_extra='', **specs_args):
    if (outdir):
        os.makedirs(outdir, exist_ok=True)
    engine = db_engine()

    variants = {
        'Non-Omicron': 'none',
        'Omicron': 'omicron'
    }

    immunities = {
        'Dose-2': {'initial_attrs': {'seir': 'S'}, 'params': {f'shot{i}_per_available': 1 for i in [1, 2]}},
        'Booster': {'initial_attrs': {'seir': 'S'}, 'params': {f'shot{i}_per_available': 1 for i in [1, 2, 3]}},
        'Prior Delta Infection': {'initial_attrs': {'seir': 'E', 'variant': 'delta'}, 'params': {}},
        'Prior Omicron Infection': {'initial_attrs': {'seir': 'E', 'variant': 'omicron'}, 'params': {}}
    }

    fig, axs = plt.subplots(2, 2)

    for (immunity_label, immunity_specs), ax in zip(immunities.items(), axs.flatten()):
        ax.set_prop_cycle(cycler(color=['paleturquoise', 'darkcyan', 'violet', 'indigo']))
        ax.set_title(f'Immunity from {immunity_label}')

        print(f'Prepping and running model for {immunity_label} immunity...')
        model = build_default_model(days, engine, base_model, **specs_args)
        for k, v in immunity_specs['params'].items():
            model.set_param(k, v)
        model.build_ode()
        model.compile()
        model.solve_ode(y0_dict={model.get_default_cmpt_by_attrs({**immunity_specs['initial_attrs'], 'age': age}): n for age, n in model.group_pops.items()})

        params = model.params_as_df
        group_by_attr_names = ['seir'] + [attr_name for attr_name in model.param_attr_names if attr_name != 'variant']
        n = model.solution_sum(group_by_attr_names).stack(level=group_by_attr_names).xs('S', level='seir')

        immunities = []
        severe_immunities = []
        for variant_label, variant in variants.items():
            if immunity_label != 'Prior Omicron Infection' or variant_label == 'Non-Omicron':
                variant_params = params.xs(variant, level='variant')

                net_severe_immunity = (n * (1 - (1 - variant_params['immunity']) * (1 - variant_params['severe_immunity']))).groupby('t').sum() / n.groupby('t').sum()
                net_severe_immunity.plot(label=f'Immunity vs Severe {"Disease" if immunity_label == "Prior Omicron Infection" else variant_label}', ax=ax)

                immunity = (n * variant_params['immunity']).groupby('t').sum() / n.groupby('t').sum()
                immunity.plot(label=f'Immunity vs {"Infection" if immunity_label == "Prior Omicron Infection" else variant_label}', ax=ax)
                immunities.append(immunity)
                severe_immunities.append(net_severe_immunity)


        ax.legend(loc='best')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(color='lightgray')
        ax.set_xlabel(f'Days Since {immunity_label}')
        ax.set_ylim((0, 1))
        ax.set_xticks(np.arange(0, days+1, 30))
        ax.set_xlim((30, days))

    fig.tight_layout()
    plt.savefig(get_filepath_prefix(outdir) + f"compartment_report_share_of_total_{fname_extra}.png")
    plt.close()


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

    parser = ModelSpecsArgumentParser()
    parser.add_argument("-d", "--days", type=int, help="the number of days of immunity to plot")
    parser.set_defaults(days=180)

    specs_args = parser.specs_args_as_dict()
    non_specs_args = parser.non_specs_args_as_dict()

    run_mock_immunity(**non_specs_args, outdir=outdir, **specs_args)
