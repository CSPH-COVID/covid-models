### Python Standard Library ###
import os
import json
import argparse
import datetime as dt
import dateutil.relativedelta as durel
### Third Party Imports ###
import pandas as pd
import numpy as np
### Local Imports ###
from covid_model import CovidModel, CovidModelSpecifications, db_engine

###################################################################################
# TODO: FIX THIS CODE; IT'S TOTALLY BROKEN
###################################################################################


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
    df['prev'] = 100000.0 * df['Itotal'] / model.model_params['total_pop']
    df['oneinX'] = model.model_params['total_pop'] / df['Itotal']
    df['Exposed'] = 100.0 * df['Einc'].cumsum()

    df.index.names = ['t']
    return df


def build_tc_df(model: CovidModel):
    return pd.DataFrame.from_dict({'time': model.tslices[:-1], 'tc_pb': model.efs, 'tc': model.obs_ef_by_slice})


def tags_to_scen_label(tags):
    if tags['run_type'] == 'Current':
        return 'Current Fit'
    elif tags['run_type'] == 'Prior':
        return 'Prior Fit'
    elif tags['run_type'] == 'Vaccination Scenario':
        return f'Vaccine Scenario: {tags["vacc_cap"]}'
    elif tags['run_type'] == 'TC Shift Projection':
        return f'TC Shift Scenario: {tags["tc_shift"]} on {tags["tc_shift_date"]} ({tags["vacc_cap"]})'


def run_model(model, engine, legacy_output_dict=None):
    print('Scenario tags: ', model.tags)
    model.solve_seir()
    model.write_results_to_db(engine, new_spec=True)
    if legacy_output_dict is not None:
        legacy_output_dict[tags_to_scen_label(model.tags)] = build_legacy_output_df(model)


def run_model_scenarios():
    if (outdir):
        os.makedirs(outdir, exist_ok=True)

    engine = db_engine()

    # set various parameters
    tmax = run_params.days if run_params.days is not None else 700
    vacc_scens = ['current trajectory', 'increased booster uptake', '75% elig. boosters by Dec 31']
    primary_vacc_scen = 'current trajectory'
    params_fname = run_params.params if run_params.params is not None else 'input/params.json'
    current_fit_id = run_params.current_fit_id if run_params.current_fit_id is not None else 1824
    prior_fit_id = run_params.prior_fit_id if run_params.prior_fit_id is not None else 1516
    tc_shifts = run_params.tc_shifts if run_params.tc_shifts is not None else [-0.05, -0.10]

    tc_shift_date = dt.date.today() + durel.relativedelta(weekday=durel.FR)
    tc_shift_days = 14
    batch = 'standard_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    # run model scenarios
    legacy_outputs = {}

    # create models for low- and high-vaccine-uptake scenarios
    model = CovidModel(end_date=(CovidModel.default_start_date + dt.timedelta(days=tmax)).date())
    specs = CovidModelSpecifications(engine=engine, from_specs=current_fit_id, new_end_date=model.end_date)
    specs.base_spec_id = current_fit_id
    specs.tags = {'batch': batch}
    vacc_proj_dict = json.load(open('input/vacc_proj_params.json'))
    for vacc_scen in vacc_scens:
        print(f'Prepping model with vaccine projection scenario "{vacc_scen}"...')
        specs.set_vacc_proj(vacc_proj_dict[vacc_scen])
        model.prep(specs=specs)
        print(f'Running scenarios...')
        model.tags.update({'run_type': 'Vaccination Scenario', 'vacc_cap': vacc_scen, 'tc_shift': 'no shift', 'tc_shift_date': 'no_shift'})
        run_model(model, engine, legacy_output_dict=legacy_outputs)
        if vacc_scen == primary_vacc_scen:
            model.tags.update({'run_type': 'Current', 'vacc_cap': vacc_scen})
            run_model(model, engine, legacy_output_dict=legacy_outputs)
            build_legacy_output_df(model).to_csv('output/out2.csv')

        base_tslices, base_tc = (model.tslices, model.tc)
        for tcs in tc_shifts:
            tc_shift_t = (tc_shift_date - model.start_date).days
            model.apply_tc(tc=base_tc + list(np.linspace(base_tc[-1], base_tc[-1] + tcs, tc_shift_days)),
                           tslices=base_tslices +  list(range(tc_shift_t, tc_shift_t + tc_shift_days)))
            model.tags.update({'run_type': 'TC Shift Projection', 'tc_shift': f'{int(100 * tcs)}%', 'tc_shift_date': tc_shift_date.strftime("%b %#d")})
            run_model(model, engine, legacy_output_dict=legacy_outputs)

    df = pd.concat(legacy_outputs)
    df.index.names = ['scenario', 'time']
    df.to_csv('output/allscenarios.csv')


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--current_fit_id", type=int, help="The current fit ID (run today)")
    parser.add_argument("-pf", "--prior_fit_id", type=int, help="The prior fit ID (run last week)")
    parser.add_argument("-d", "--days", type=int, help="Number of days to include in these scenarios, starting from Jan 24, 2020")
    parser.add_argument("-tcs", "--tc_shifts", nargs='+', type=float, help="Upcoming shifts in TC to simulate (-0.05 represents a 5% reduction in TC)")
    parser.add_argument("-p", "--params", type=str, help="the path to the params file to use for fitting; default to 'input/params.json'")
    # parser.add_argument("-pvs", "--primary_vaccine_scen", choices=['high', 'low'], type=float, help="The name of the vaccine scenario to be used for the default model scenario.")

    run_model_scenarios()