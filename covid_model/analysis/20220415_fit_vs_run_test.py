### Python Standard Library ###
import os
import datetime as dt
import json
### Third Party Imports ###
import numpy as np
import pandas as pd
### Local Imports ###
from covid_model.run_scripts import run_solve_seir, run_fit
from covid_model.utils import get_file_prefix
from covid_model import CovidModel


def main():
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))
    os.makedirs(outdir, exist_ok=True)

    fit_args = {
        'batch_size': 6,
        'increment_size': 2,
        'tc_min': -0.99,
        'tc_max': 0.99,
        'forward_sim_each_batch': True,
        'multiprocess': 2,
        'look_back': None,
        'use_base_specs_end_date': True,
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
        'regions': ['ms', 'cent'],
        'start_date': dt.datetime.strptime('2020-02-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-04-01', "%Y-%m-%d").date(),
        'tc': [0.75, 0.75],
        'tslices': [14],
        'max_step_size': 0.5
    }
    with open(get_file_prefix(outdir) + "________________________.txt", 'w') as f:
        f.write(json.dumps({"fit_args": fit_args, "spec_args": specs_args}, default=str, indent=4))


    mob_modes = ['none']
    #mob_modes = ['none', 'location_attached']

    print('Run fit')
    #write_infos = run_fit(**fit_args, **run_args, outdir=outdir)
    ms = run_fit(**fit_args, **specs_args, outdir=outdir)
    specs_args['region_fit_spec_ids'] = [x['specs_info']['spec_id'] if isinstance(x, dict) else x.spec_id for x in ms]
    #run_args['region_fit_spec_ids'] = [2287, 2288]


    dfs = []
    dfhs = []
    ms2 = []
    for mm in mob_modes:
        print(f'Mobility Mode: {mm}')
        print('Run Forward Sim')
        specs_args.update({'mobility_mode': mm, 'refresh_actual_mobility': None if mm == 'none' else True})
        model, df, dfh = run_solve_seir(**specs_args, outdir=outdir)
        ms2.append(model)
        dfs.append(df.assign(mobility=mm))
        dfhs.append(dfh.assign(mobility=mm))
    df = pd.concat(dfs).reset_index(drop=True)
    dfh = pd.concat(dfhs, axis=0)


    print("saving results")
    df.to_csv(get_file_prefix(outdir) + "fit_vs_run_run_compartments.csv")
    dfh.to_csv(get_file_prefix(outdir) + "fit_vs_run_run_forwardsim_hospitalized.csv")

    #print("plotting results")
    #p = sns.relplot(data=df, x='date', y='y', hue='mobility', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    #_ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    #plt.savefig(get_file_prefix(outdir) + "fit_vs_run_test_compartments.png", dpi=300)

    #p = sns.relplot(data=dfh, x='date', y='hospitalized', hue='mode', col='region', col_wrap=min(3, len(run_args['regions'])), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    #_ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    #plt.savefig(get_file_prefix(outdir) + "fit_vs_run_test_hospitalized.png", dpi=300)

    # compare DEQ components from the disconnected models to the connected model
    # only possible if we have the model instances
    if isinstance(ms[0], CovidModel):
        idxs = [[ms2[0].cmpt_idx_lookup[cmpt] for cmpt in m.compartments] for m in ms]
        comparisons = [None] * len(ms)
        for i, tup in enumerate(zip(ms, idxs)):
            comparisons[i] = {}
            m, idx = tup
            for t in m.trange:
                comparisons[i][t] = {}
                comparisons[i][t]['const_vector'] = ([ms2[0].constant_vector[t][idx]] == m.constant_vector[t]).all()
                comparisons[i][t]['linear_matrix'] = (ms2[0].linear_matrix[t].toarray()[idx, :][:, idx] == m.linear_matrix[t].toarray()).all()
                comparisons[i][t]['nonlinear_matrices'] = []
                for k in m.nonlinear_matrices[t].keys():
                    k2 = tuple(idx[x] for x in k)
                    comparisons[i][t]['nonlinear_matrices'].append(np.allclose(ms2[0].nonlinear_matrices[t][k2].toarray()[idx, :][:, idx] * ms2[0].nonlinear_multiplier[t], m.nonlinear_matrices[t][k].toarray() * m.nonlinear_multiplier[t]))

        print([v['const_vector'] for c in comparisons for v in c.values()])
        print([v['linear_matrix'] for c in comparisons for v in c.values()])
        print([tv for c in comparisons for v in c.values() for tv in v['nonlinear_matrices']])

    print("done")


if __name__ == "__main__":
    main()