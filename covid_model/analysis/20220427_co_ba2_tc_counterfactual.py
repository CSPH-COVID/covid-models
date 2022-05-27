### Python Standard Library ###
import os
import datetime as dt
import json
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
### Local Imports ###
from covid_model import CovidModel
from covid_model.run_scripts import run_compartment_report, run_solve_seir
from covid_model.utils import get_filepath_prefix
from covid_model.db import db_engine


def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    basename = os.path.basename(__file__)
    fname_prefix = basename.split(".")[0]
    outdir = os.path.join("covid_model", "output", basename)
    os.makedirs(outdir, exist_ok=True)

    specs_args = {
        'regions': ['co'],
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-08-01', "%Y-%m-%d").date(),
        'max_step_size': None,
        'from_specs': 2378
    }
    with open(get_filepath_prefix(outdir) + "________________________.txt", 'w') as f:
        f.write(json.dumps({"specs_args": specs_args}, default=str, indent=4))

    from_date = dt.datetime.strptime('2021-10-01', "%Y-%m-%d")
    to_date = dt.datetime.strptime('2022-08-01', "%Y-%m-%d")



    ####################################################################################################################
    # Run

    # fit a statewide model up to present day to act as a baseline
    engine = db_engine()
    print("creating model")
    model = CovidModel(engine=engine, **specs_args)
    m0, df0, dfh0 = run_solve_seir(outdir=outdir, fname_extra="as_fit", model=model)
    replace_idx = [i for i, tc in enumerate(model.tc) if tc > 0][-1] + 1
    new_tc = [model.tc[replace_idx-1]] * (len(model.tc)-replace_idx)
    model.apply_tc(tc=new_tc)
    m1, df1, dfh1 = run_solve_seir(outdir=outdir, fname_extra="tc_const", model=model)


    print("running compartment report")
    run_compartment_report(from_date, to_date, group_by_attr_names=['age', 'vacc', 'priorinf', 'immun', 'seir', 'variant'], model=m0, outdir=outdir, fname_extra="default")
    run_compartment_report(from_date, to_date, group_by_attr_names=['age', 'vacc', 'priorinf', 'immun', 'seir', 'variant'], model=m1, outdir=outdir, fname_extra="tc_const")

    print('Plotting')
    df = pd.concat([df0.assign(scen='as_fit'), df1.assign(scen='tc_const')])
    dfh = pd.concat([dfh0.assign(scen='as_fit'), dfh1.assign(scen='tc_const')])
    df = df.set_index(['date', 'region', 'seir', 'scen'])
    dfh = dfh.set_index(['scen'], append=True)
    dfh = dfh.melt(value_vars=['modeled_hospitalized', 'currently_hospitalized'], ignore_index=False, var_name='series', value_name='val').set_index('series', append=True)

    p = sns.relplot(data=df.loc[df.index.get_level_values('date') >= from_date], x='date', y='y', hue='scen', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=3, aspect=3)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + fname_prefix + "_compartments.png", dpi=300)

    p = sns.relplot(data=dfh.loc[dfh.index.get_level_values('date') >= from_date], x='date', y='val', hue='series', col='region', row='scen', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=3, aspect=3)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + fname_prefix + "_hospitalized.png", dpi=300)

    # run compartment reports on each scenario

    print("done")


if __name__ == "__main__":
    main()