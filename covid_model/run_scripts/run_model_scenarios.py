### Python Standard Library ###
from operator import attrgetter
import os
import json
from datetime import date
import datetime as dt
from multiprocessing import Pool
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
### Local Imports ###
from covid_model import CovidModel, ModelSpecsArgumentParser, db_engine, CovidModelFit, CovidModelSpecifications
from covid_model.run_scripts.run_solve_seir import run_solve_seir
from covid_model.run_scripts.run_compartment_report import run_compartment_report
from covid_model.run_scripts.run_omicron_report import run_omicron_report
from covid_model.utils import get_filepath_prefix


def build_legacy_output_df(model: CovidModel):
    ydf = model.solution_sum(['seir', 'age']).stack(level='age')
    dfs_by_group = []
    for i, group in enumerate(model.attr['age']):
        dfs_by_group.append(ydf.xs(group, level='age').rename(columns={var: var + str(i+1) for var in model.attr['seir']}))
    df = pd.concat(dfs_by_group, axis=1)

    params_df = model.params_as_df
    combined = model.solution_ydf.stack(model.param_attr_names).join(params_df)

    totals = model.solution_sum('seir')
    totals_by_priorinf = model.solution_sum(['seir', 'priorinf'])
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
    df['prev'] = 100000.0 * df['Itotal'] / model.model_params['total_pop'][0]['values']
    df['oneinX'] = model.model_params['total_pop'][0]['values'] / df['Itotal']
    df['Exposed'] = 100.0 * df['Einc'].cumsum()

    df.index.names = ['t']
    return df


def run_single_scenario(args):
    engine = db_engine()
    scen, outdir, base_specs, scens_files, refit_from_date, fit_args, plot_from_date, plot_to_date, return_model = args
    print(f"Scenario: {scen}: Copying / Modifying Model")
    scen_base_model = CovidModel(engine=engine, **base_specs)
    # Update params based on scenario
    if 'params_scens' in scens_files.keys():
        scen_base_model.model_params.update(scens_files['params_scens'])
    if 'vacc_proj_params_scens' in scens_files.keys():
        scen_base_model.vacc_proj_params.update(scens_files['vacc_proj_params_scens'])
    if 'mobility_proj_params_scens' in scens_files.keys():
        scen_base_model.mobility_proj_params.update(scens_files['mobility_proj_params_scens'])
    # Note: attribute multipliers is a list, so our easiest option is to append. So you probably want to remove
    # something from the base attribute multipliers file if the scenarios are exploring different settings for those.
    if 'attribute_multipliers_scens' in scens_files.keys():
        scen_base_model.attribute_multipliers.extend(scens_files['attribute_multipliers_scens'])
    scen_model = CovidModel(base_model=scen_base_model)
    if refit_from_date is not None:
        fit_args['look_back'] = len(
            [t for t in scen_model.tslices if t >= (refit_from_date - scen_model.start_date).days])
        fit = CovidModelFit(engine=engine, tc_min=fit_args['tc_min'], tc_max=fit_args['tc_max'], from_specs=scen_model)
        fit.set_actual_hosp(engine=engine, county_ids=scen_model.get_all_county_fips())
        fit.run(engine, **fit_args, print_prefix=f'{scen}', outdir=outdir)
        scen_model.apply_tc(fit.fitted_model.tc, fit.fitted_model.tslices)
    print(f"Scenario: {scen}: Prepping and Solving SEIR")
    scen_model, df, dfh = run_solve_seir(outdir=outdir, model=scen_model, prep_model=True, write_model_to_db=return_model, tags={'scenario': scen}, fname_extra=scen)
    dflo = build_legacy_output_df(scen_model)
    print("running omicron report")
    run_omicron_report(plots=['prev', 'hosp','var','imm'], from_date=dt.datetime.strptime('2021-07-01', "%Y-%m-%d").date(), model=scen_model, outdir=outdir, fname_extra=scen)
    print("running compartment report")
    run_compartment_report(plot_from_date, plot_to_date, group_by_attr_names=['age', 'vacc', 'priorinf', 'immun', 'seir', 'variant'], model=scen_model, outdir=outdir, fname_extra=scen)
    scen_model.solution_sum(index_with_model_dates=True).unstack().to_csv(get_filepath_prefix(outdir) + f"{scen}.csv")


    if return_model:
        return {'df': df.assign(scen=scen), 'dfh': dfh.assign(scen=scen), 'dflo': dflo.assign(scen=scen), 'model': scen_model}
    else:
        return {'df': df.assign(scen=scen), 'dfh': dfh.assign(scen=scen), 'dflo': dflo.assign(scen=scen), 'specs_info': scen_model.prepare_write_specs_query(), 'results_info': scen_model.prepare_write_results_query()}


def run_model_scenarios(params_scens, vacc_proj_params_scens, mobility_proj_params_scens,
                        attribute_multipliers_scens, outdir, plot_from_date, plot_to_date, fname_extra="",
                        refit_from_date=None, fit_args=None, multiprocess=None, **specs_args):
    if (outdir):
        os.makedirs(outdir, exist_ok=True)
    engine = db_engine()

    # compile scenarios:
    locs = [key for key, val in locals().items() if val is not None]
    scen_args_str = [s for s in ['params_scens', 'vacc_proj_params_scens', 'mobility_proj_params_scens', 'attribute_multipliers_scens'] if s in locs]
    scens_files = {}
    for sf in scen_args_str:
        scen_dict = json.load(open(eval(sf), 'r'))
        for scen, scen_settings in scen_dict.items():
            if scen not in scens_files.keys():
                scens_files[scen] = {}
            scens_files[scen].update({sf: scen_settings})

    scens = list(scens_files.keys())

    # initialize Base model:
    base_model = CovidModel(engine=engine, **specs_args)

    if multiprocess:
        args_list = map(lambda x: [x, outdir, specs_args, scens_files[x], refit_from_date, fit_args, plot_from_date, plot_to_date, False], scens)
        p = Pool(multiprocess)
        results = p.map(run_single_scenario, args_list)
        # write results to database serially
        for i, _ in enumerate(results):
            results[i]['specs_info'] = CovidModelSpecifications.write_prepared_specs_to_db(results[i]['specs_info'], db_engine())
            results[i]['results_info'] = CovidModel.write_prepared_results_to_db(results[i]['results_info'], db_engine(), results[i]['specs_info']['spec_id'])
    else:
        args_list = map(lambda x: [x, outdir, specs_args, scens_files[x], refit_from_date, fit_args, plot_from_date, plot_to_date, True], scens)
        results = list(map(run_single_scenario, args_list))
        # results already written to db

    df = pd.concat([r['df'] for r in results], axis=0).set_index(['scen', 'date', 'region', 'seir'])
    dfh = pd.concat([r['dfh'] for r in results], axis=0)
    dflo = pd.concat([r['dflo'] for r in results], axis=0)

    #df = pd.concat(dfs, axis=0).set_index(['scen', 'date', 'region', 'seir'])
    #dfh = pd.concat(dfhs, axis=0)
    dfh_measured = dfh[['currently_hospitalized']].rename(columns={'currently_hospitalized': 'hospitalized'}).loc[dfh['scen'] == scens[0]].assign(series='observed')
    dfh_modeled = dfh[['modeled_hospitalized', 'scen']].rename(columns={'modeled_hospitalized': 'hospitalized', 'scen': 'series'})
    dfh2 = pd.concat([dfh_measured, dfh_modeled], axis=0).set_index('series', append=True)

    print("saving results")
    df.to_csv(get_filepath_prefix(outdir) + f"run_model_scenarios_compartments_{fname_extra}.csv")
    dfh.to_csv(get_filepath_prefix(outdir) + f"run_model_scenarios_hospitalized_{fname_extra}.csv")
    dfh2.to_csv(get_filepath_prefix(outdir) + f"run_model_scenarios_hospitalized2_{fname_extra}.csv")
    dflo.to_csv(get_filepath_prefix(outdir) + f"out2_{fname_extra}.csv")

    print("plotting results")
    p = sns.relplot(data=df, x='date', y='y', hue='scen', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + f"run_model_scenarios_compartments_{fname_extra}.png", dpi=300)

    p = sns.relplot(data=dfh2, x='date', y='hospitalized', hue='series', col='region', col_wrap=min(3, len(specs_args['regions'])), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + f"run_model_scenarios_hospitalized_{fname_extra}.png", dpi=300)

    print("done")

    return df, dfh, dfh2


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

    parser = ModelSpecsArgumentParser()
    parser.add_argument("-psc", '--params_scens', type=str, help="path to parameters scenario file to use (updates base model parameters)")
    parser.add_argument("-vppsc", '--vacc_proj_params_scens', type=str, help="path to vaccine projection parameters scenario file (updates base vpp)")
    parser.add_argument("-mppsc", '--mobility_proj_params_scens', type=str, help="path to mobility projection parameters scenario file (updates base mpp)")
    parser.add_argument('-amsc', '--attribute_multipliers_scens', type=str, help="path to attribute multipliers scenario file (updates base mprev)")
    parser.add_argument('-rfd', '--refit_from_date', type=date.fromisoformat, help="refit from this date forward for each scenario, or don't refit if None (format: YYYY-MM-DD)")
    parser.add_argument("-fne", '--fname_extra', default="", help="extra info to add to all files saved to disk")

    specs_args = parser.specs_args_as_dict()
    non_specs_args = parser.non_specs_args_as_dict()

    # note refitting doesn't work from CLI because we aren't collecting fit specs here. Better way to do this?

    run_model_scenarios(**non_specs_args, outdir=outdir, **specs_args)
