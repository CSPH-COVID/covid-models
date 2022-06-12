### Python Standard Library ###
from operator import attrgetter
import os
import json
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
### Local Imports ###
from covid_model import CovidModel, ModelSpecsArgumentParser, db_engine
from covid_model.run_scripts.run_solve_seir import run_solve_seir
from covid_model.utils import get_filepath_prefix

###################################################################################
# TODO: FIX THIS CODE; IT'S TOTALLY BROKEN
###################################################################################


def build_legacy_output_df(model: CovidModel):
    ydf = model.solution_sum_df(['seir', 'age']).stack(level='age')
    dfs_by_group = []
    for i, group in enumerate(model.attrs['age']):
        dfs_by_group.append(ydf.xs(group, level='age').rename(columns={var: var + str(i+1) for var in model.attrs['seir']}))
    df = pd.concat(dfs_by_group, axis=1)

    params_df = model.params_as_df
    combined = model.solution_ydf.stack(model.param_attr_names).join(params_df)

    totals = model.solution_sum_df('seir')
    totals_by_priorinf = model.solution_sum_df(['seir', 'priorinf'])
    df['Iht'] = totals['Ih']
    df['Dt'] = totals['D']
    df['Rt'] = totals_by_priorinf[('S', 'none')]
    df['Itotal'] = totals['I'] + totals['A']
    df['Etotal'] = totals['E']
    df['Einc'] = (combined['E'] / combined['alpha']).groupby('t').sum()
    # df['Einc'] = totals_by_variant * params_df / model.model_params['alpha']
    # for i, age in enumerate(model.attr['age']):
    #     df[f'Vt{i+1}'] = (model.solution_ydf[('S', age, 'vacc')] + model.solution_ydf[('R', age, 'vacc')]) * params_df.xs((age, 'vacc'), level=('age', 'vacc'))['vacc_eff']
    #     df[f'immune{i+1}'] = by_age[('R', age)] + by_age_by_vacc[('S', age, 'vacc')] * params_df.xs((age, 'vacc'), level=('age', 'vacc'))['vacc_eff']
    df['Vt'] = model.immunity(variant='omicron', vacc_only=True)
    df['immune'] = model.immunity(variant='omicron')
    df['date'] = model.daterange
    df['Ilag'] = totals['I'].shift(3)
    df['Re'] = model.re_estimates
    df['prev'] = 100000.0 * df['Itotal'] / model.model_params['total_pop']
    df['oneinX'] = model.model_params['total_pop'] / df['Itotal']
    df['Exposed'] = 100.0 * df['Einc'].cumsum()

    df.index.names = ['t']
    return df

def run_model_scenarios(params_scens, vacc_proj_params_scens, mobility_proj_params_scens,
                        attribute_multipliers_scens, outdir, **specs_args):
    if (outdir):
        os.makedirs(outdir, exist_ok=True)
    engine = db_engine()

    # compile scenarios:
    scens_files = [json.load(open(sf, 'r')) if sf is not None else None for sf in [params_scens, vacc_proj_params_scens, mobility_proj_params_scens, attribute_multipliers_scens]]
    scens = [key for sf in scens_files if sf is not None for key in sf.keys()]

    # initialize Base model:
    base_model = CovidModel(engine=engine, **specs_args)

    ms = []
    dfs = []
    dfhs = []
    for scen in scens:
        print(f"Scenario: {scen}: Copying / Modifying Model")
        scen_base_model = CovidModel(engine=engine, base_model=base_model)
        # Update params based on scenario
        if scens_files[0] and scen in scens_files[0]:
            scen_base_model.params_by_t.update(scens_files[0][scen])
        if scens_files[1] and scen in scens_files[1]:
            scen_base_model.vacc_proj_params.update(scens_files[1][scen])
        if scens_files[2] and scen in scens_files[2]:
            scen_base_model.mobility_proj_params.update(scens_files[2][scen])
        # Note: attribute multipliers is a list, so our easiest option is to append. So you probably want to remove
        # something from the base attribute multipliers file if the scenarios are exploring different settings for those.
        if scens_files[3] and scen in scens_files[3]:
            scen_base_model.attribute_multipliers.extend(scens_files[3][scen])
        scen_model = CovidModel(base_model=scen_base_model)
        print(f"Scenario: {scen}: Prepping and Solving SEIR")
        scen_model, df, dfh = run_solve_seir(outdir=outdir, model=scen_model, tags={'scenario': scen})
        ms.append(scen_model)
        dfs.append(df.assign(scen=scen))
        dfhs.append(dfh.assign(scen=scen))

    df = pd.concat(dfs, axis=0)
    dfh = pd.concat(dfhs, axis=0)
    dfh_measured = dfh[['currently_hospitalized']].rename(columns={'currently_hospitalized': 'hospitalized'}).loc[dfh['scen'] == scens[0]].assign(series='observed')
    dfh_modeled = dfh[['modeled_hospitalized', 'scen']].rename(columns={'modeled_hospitalized': 'hospitalized', 'scen': 'scen'})
    dfh2 = pd.concat([dfh_measured, dfh_modeled], axis=0).set_index('series', append=True)

    print("saving results")
    df.to_csv(get_filepath_prefix(outdir) + "run_model_scenarios_compartments.csv")
    dfh.to_csv(get_filepath_prefix(outdir) + "run_model_scenarios_hospitalized.csv")
    dfh2.to_csv(get_filepath_prefix(outdir) + "run_model_scenarios_hospitalized2.csv")

    print("plotting results")
    p = sns.relplot(data=df, x='date', y='y', hue='scen', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + "run_model_scenarios_compartments.png", dpi=300)


    p = sns.relplot(data=dfh2, x='date', y='hospitalized', hue='scen', col='region', col_wrap=min(3, len(specs_args['regions'])), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + "run_model_scenarios_hospitalized.png", dpi=300)

    print("done")

    return(df, dfh, dfh2, ms)


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

    parser = ModelSpecsArgumentParser()
    parser.add_argument("-psc", '--params_scens', type=str, help="path to parameters scenario file to use (updates base model parameters)")
    parser.add_argument("-vppsc", '--vacc_proj_params_scens', type=str, help="path to vaccine projection parameters scenario file (updates base vpp)")
    parser.add_argument("-mppsc", '--mobility_proj_params_scens', type=str, help="path to mobility projection parameters scenario file (updates base mpp)")
    parser.add_argument('-amsc', '--attribute_multipliers_scens', type=str, help="path to attribute multipliers scenario file (updates base mprev)")

    specs_args = parser.specs_args_as_dict()
    non_specs_args = parser.non_specs_args_as_dict()

    run_model_scenarios(**non_specs_args, outdir=outdir, **specs_args)
