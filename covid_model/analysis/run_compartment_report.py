import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime as dt
from covid_model.analysis.charts import modeled

from covid_model.db import db_engine
import covid_model
from covid_model import CovidModel, RegionalCovidModel, all_regions
from covid_model.cli_specs import ModelSpecsArgumentParser

if __name__ == '__main__':
    engine = db_engine()
    argparser = ModelSpecsArgumentParser()
    argparser.add_argument('-fd', '--from_date', type=dt.date.fromisoformat, default=dt.date(2020, 3, 1), help='x-axis minimum date')
    argparser.add_argument('-td', '--to_date', type=dt.date.fromisoformat, default=dt.date.today(), help='x-axis maximum date')
    argparser.add_argument("-mc", "--model_class", default="CovidModel", help="Name of the class to use for the model (Defaults to CovidModel)")
    argparser.add_argument("-rg", "--region", choices=all_regions.keys(), required=False, help="Specify the region to be run, if running a regional model, if not specified, just runs default parameters")
    argparser.add_argument("-sp", "--save_prefix", default=None, help="If specified, saves the figure with the given prefix")
    argparser.add_argument('-gba', '--group_by_attr_names', nargs='+', type=str,
                           choices=CovidModel.attr.keys(), default=['age', 'vacc', 'priorinf'],
                           help=f'list of attributes name to split charts by')

    args = argparser.parse_args()
    model_class = getattr(covid_model, args.model_class)
    model_args = {"region": args.region} if model_class == RegionalCovidModel else {}
    model = model_class(engine=engine, **model_args, **argparser.specs_args_as_dict())
    model.prep()
    model.solve_seir()

    seir = {'susceptible': 'S', 'infected': ['I', 'A'], 'hospitalized': 'Ih'}
    fig, axs = plt.subplots(len(seir), len(args.group_by_attr_names), figsize=(5*len(seir), 5*len(args.group_by_attr_names)), dpi=300)

    for i, (seir_label, seir_attrs) in enumerate(seir.items()):
        for j, attr_name in enumerate(args.group_by_attr_names):
            ax = axs[i, j]
            modeled(model, seir_attrs, groupby=attr_name, share_of_total=True, ax=ax)
            ax.set_xlim(args.from_date, args.to_date)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylabel(f'{seir_label} by {attr_name}')
            ax.legend(loc='upper left')

    fig.tight_layout()
    if args.save_prefix:
        plt.savefig("output/" + args.save_prefix + "_compartment_report_grpby_" + "_".join(args.group_by_attr_names) + ".png")
    else:
        plt.show()
