### Python Standard Library ###
import os
import datetime as dt
### Third Party Imports ###
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
### Local Imports ###
from covid_model.analysis.charts import modeled
from covid_model.db import db_engine
import covid_model
from covid_model import CovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser
from covid_model.utils import get_file_prefix


def run_compartment_report(from_date, to_date, group_by_attr_names: list, outdir, model=None, **specs_args):
    if model is None:
        model = CovidModel(engine=engine, **specs_args)
        model.prep()
        model.solve_seir()

    seir = {'susceptible': 'S', 'infected': ['I', 'A'], 'hospitalized': 'Ih'}
    fig, axs = plt.subplots(len(seir), len(group_by_attr_names), figsize=(7*len(group_by_attr_names), 5*len(seir)), dpi=300)

    for i, (seir_label, seir_attrs) in enumerate(seir.items()):
        for j, attr_name in enumerate(group_by_attr_names):
            ax = axs[i, j]
            modeled(model, seir_attrs, groupby=attr_name, share_of_total=True, ax=ax)
            ax.set_xlim(from_date, to_date)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylabel(f'{seir_label} by {attr_name}')
            ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig(get_file_prefix(outdir) + "compartment_report_" + "_".join(group_by_attr_names) + "_share_of_total.png")
    plt.close()

    # repeat but don't normalize to a percentage.
    fig, axs = plt.subplots(len(seir), len(group_by_attr_names), figsize=(7*len(group_by_attr_names), 5*len(seir)), dpi=300)
    for i, (seir_label, seir_attrs) in enumerate(seir.items()):
        for j, attr_name in enumerate(group_by_attr_names):
            ax = axs[i, j]
            modeled(model, seir_attrs, groupby=attr_name, share_of_total=False, ax=ax)
            ax.set_xlim(from_date, to_date)
            ax.set_ylabel(f'{seir_label} by {attr_name}')
            ax.legend(loc='upper left')

    fig.tight_layout()
    plt.savefig(get_file_prefix(outdir) + "compartment_report_" + "_".join(group_by_attr_names) + ".png")
    plt.close()


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

    parser = ModelSpecsArgumentParser()

    engine = db_engine()
    parser.add_argument('-fd', '--from_date', type=dt.date.fromisoformat, default=dt.date(2020, 3, 1), help='x-axis minimum date')
    parser.add_argument('-td', '--to_date', type=dt.date.fromisoformat, default=dt.date.today(), help='x-axis maximum date')
    parser.add_argument('-gba', '--group_by_attr_names', nargs='+', type=str, choices=CovidModel.attr.keys(), default=['age', 'vacc', 'priorinf'], help=f'list of attributes name to split charts by')

    specs_args = parser.specs_args_as_dict()
    non_specs_args = parser.non_specs_args_as_dict()


    run_compartment_report(**specs_args, **non_specs_args, outdir=outdir)