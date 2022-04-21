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
from covid_model.run_scripts import run_solve_seir, run_fit
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
        'bsed': True,
        'multiprocess': 2,
        'look_back': None,
        'use_base_specs_end_date': True,
        'window_size': 14,
        'write_batch_output': False
    }
    run_args = {
        'params': 'covid_model/input/params.json',
        'region_definitions': 'covid_model/input/region_definitions.json',
        'timeseries_effect_multipliers': 'covid_model/input/timeseries_effects/multipliers.json',
        'mab_prevalence': 'covid_model/input/timeseries_effects/mab_prevalence.csv',
        'attribute_multipliers': 'covid_model/input/attribute_multipliers.json',
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',
        'refresh_actual_vacc': True,
        'refrech_actual_mobility': True,
        'regions': ['ms', 'cent'],
        'start_date': dt.datetime.strptime('2020-03-01', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-04-01', "%Y-%m-%d").date(),
        'tc': [0.75, 0.75],
        'tslices': [14],
        'max_step_size': 0.5
    }
    with open(get_file_prefix(outdir) + "________________________.txt", 'w') as f:
        f.write(json.dumps(fit_args, default=str, indent=4))
        f.write(json.dumps(run_args, default=str, indent=4))


    ####################################################################################################################
    # Run
    mob_modes = ['population_attached', 'none']

    print('Run fit')
    ms = run_fit(**fit_args, **run_args, outdir=outdir)
    run_args['region_fit_spec_ids'] = [x['specs_info']['spec_id'] if isinstance(x, dict) else x.spec_id for x in ms]
    run_args['region_fit_result_ids'] = [x['results_info']['result_id'].iloc[0].item() if isinstance(x, dict) else x.result_id for x in ms]

    #run_args['region_fit_spec_ids'] = [2214, 2215]
    #run_args['region_fit_result_ids'] = [144, 145]

    dfs = []
    dfhs = []
    ms2 = []
    for mm in mob_modes:
        print(f'Mobility Mode: {mm}')
        print('Run Forward Sim')
        run_args.update({'mobility_mode': mm, 'refresh_actual_mobility': None if mm == 'none' else True})
        model, df, dfh = run_solve_seir(**run_args, outdir=outdir)
        ms2.append(model)
        dfs.append(df.assign(mobility=mm))
        dfhs.append(dfh.assign(mobility=mm))
        with open(get_file_prefix(outdir) + f'{mm}_ode_terms.json', 'w') as f:
            f.write(model.ode_terms_as_json())
    df = pd.concat(dfs).reset_index(drop=True)
    dfh = pd.concat(dfhs, axis=0)


    print("saving results")
    df.to_csv(get_file_prefix(outdir) + "disconnected_vs_connected_run_compartments.csv")
    dfh.to_csv(get_file_prefix(outdir) + "disconnected_vs_connected_run_forwardsim_hospitalized.csv")

    print("plotting results")
    p = sns.relplot(data=df, x='date', y='y', hue='mobility', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_file_prefix(outdir) + "disconnected_vs_connected_test_compartments.png", dpi=300)

    dfh_measured = dfh[['currently_hospitalized']].rename(columns={'currently_hospitalized': 'hospitalized'}).loc[dfh['mobility'] == 'none'].assign(series='observed')
    dfh_modeled = dfh[['modeled_hospitalized', 'mobility']].rename(columns={'modeled_hospitalized': 'hospitalized', 'mobility': 'series'})
    dfh2 = pd.concat([dfh_measured,dfh_modeled], axis=0).set_index('series', append=True)
    p = sns.relplot(data=dfh2, x='date', y='hospitalized', hue='series', col='region', col_wrap=min(3, len(run_args['regions'])), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_file_prefix(outdir) + "disconnected_vs_connected_test_hospitalized.png", dpi=300)

    print("done")


if __name__ == "__main__":
    main()