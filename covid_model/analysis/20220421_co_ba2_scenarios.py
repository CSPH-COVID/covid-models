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
from covid_model.run_scripts import run_fit, run_model_scenarios, run_compartment_report
from covid_model.utils import get_filepath_prefix


def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))
    os.makedirs(outdir, exist_ok=True)

    fit_args = {
        'batch_size': 6,
        'increment_size': 2,
        'tc_min': -0.99,
        'tc_max': 0.99,
        'forward_sim_each_batch': True,
        'multiprocess': None,
        'look_back': None,
        'use_base_specs_end_date': False,
        'window_size': 14,
        'write_batch_output': False
    }
    specs_args = {
        'params': 'covid_model/input/params.json',
        'region_definitions': 'covid_model/input/region_definitions.json',
        'timeseries_effect_multipliers': 'covid_model/input/timeseries_effects/multipliers.json',
        'mab_prevalence': 'covid_model/input/timeseries_effects/mab_prevalence.csv',
        'attribute_multipliers': 'covid_model/input/attribute_multipliers.json',
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',
        'refresh_actual_vacc': True,
        'refrech_actual_mobility': True,
        'regions': ['co'],
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-08-01', "%Y-%m-%d").date(),
        'max_step_size': 0.5
    }
    scen_args = {
        'params_scens': None,
        'vacc_proj_params_scens': None,
        'mobility_proj_params_scens': None,
        'timeseries_effect_multipliers_scens': None,
        'mab_prevalence_scens': None,
        'attribute_multipliers_scens': 'covid_model/input/20220421_ba2_scenarios/attribute_multipliers_scens_ba2.json'
    }
    with open(get_filepath_prefix(outdir) + "________________________.txt", 'w') as f:
        f.write(json.dumps({"fit_args": fit_args, "specs_args": specs_args, "scen_args": scen_args}, default=str, indent=4))

    from_date = dt.datetime.strptime('2021-10-01', "%Y-%m-%d")
    to_date = dt.datetime.strptime('2022-08-01', "%Y-%m-%d")



    ####################################################################################################################
    # Run

    # fit a statewide model up to present day to act as a baseline
    print('Run fit')
    #model = run_fit(**fit_args, outdir=outdir, **{'tc':[0.75,0.75], 'tslices':[14], **specs_args})[0]
    #specs_args["from_specs"] = model.spec_id
    specs_args["from_specs"] = 2339
    print('Run Scenarios')
    df, dfh, dfh2, ms = run_model_scenarios(**specs_args, **scen_args, outdir=outdir)

    print("running compartment report")
    [run_compartment_report(from_date, to_date, group_by_attr_names=['age', 'vacc', 'priorinf', 'immun', 'seir', 'variant'], model=m, outdir=outdir) for m in ms]

    print('Plotting')

    p = sns.relplot(data=df.loc[df.index.get_level_values('date') >= from_date], x='date', y='y', hue='scen', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=3, aspect=3)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + "ba2_scenarios_compartments.png", dpi=300)


    p = sns.relplot(data=df.loc[df.index.get_level_values('date') >= from_date], x='date', y='y', hue='scen', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=3, aspect=3)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + "ba2_scenarios_compartments.png", dpi=300)

    p = sns.relplot(data=dfh2.loc[dfh2.index.get_level_values('date') >= from_date], x='date', y='hospitalized', hue='series', col='region', col_wrap=min(3, len(specs_args['regions'])), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=3)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + "ba2_scenarios_hospitalized.png", dpi=300)

    # run compartment reports on each scenario

    print("done")


if __name__ == "__main__":
    main()