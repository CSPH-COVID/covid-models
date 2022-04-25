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
from covid_model.run_scripts import run_fit, run_model_scenarios
from covid_model.utils import get_file_prefix


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
    with open(get_file_prefix(outdir) + "________________________.txt", 'w') as f:
        f.write(json.dumps({"fit_args": fit_args, "specs_args": specs_args, "scen_args": scen_args}, default=str, indent=4))



    ####################################################################################################################
    # Run

    # fit a statewide model up to present day to act as a baseline
    print('Run fit')
    #model = run_fit(**fit_args, {'tc':[0.75,0.75], 'tslices':[14], **specs_args}, outdir=outdir)[0]
    #specs_args["from_specs"] = model.spec_id
    specs_args["from_specs"] = 2295
    print('Run Scenarios')
    df, dfh, dfh2, ms = run_model_scenarios(**specs_args, **scen_args, outdir=outdir)

    print('Plotting')
    from_date = dt.datetime.strptime('2021-10-01', "%Y-%m-%d").date(),
    p = sns.relplot(data=df, x='date', y='y', hue='scen', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_file_prefix(outdir) + "run_model_scenarios_compartments.png", dpi=300)

    dfh_measured = dfh[['currently_hospitalized']].rename(columns={'currently_hospitalized': 'hospitalized'}).loc[dfh['scen'] == scens[0]].assign(series='observed')
    dfh_modeled = dfh[['modeled_hospitalized', 'scen']].rename(columns={'modeled_hospitalized': 'hospitalized', 'scen': 'scen'})
    dfh2 = pd.concat([dfh_measured, dfh_modeled], axis=0).set_index('series', append=True)
    p = sns.relplot(data=dfh2, x='date', y='hospitalized', hue='scen', col='region', col_wrap=min(3, len(specs_args['regions'])), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_file_prefix(outdir) + "run_model_scenarios_hospitalized.png", dpi=300)

    print("done")


if __name__ == "__main__":
    main()