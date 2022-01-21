from covid_model.analysis.charts import actual_hosps, modeled
from covid_model.model_specs import CovidModelSpecifications
from model import CovidModel
from db import db_engine
import datetime as dt
import dateutil.relativedelta as durel
import pandas as pd
import numpy as np
import json
import argparse

import matplotlib.pyplot as plt


def build_legacy_output_df(model: CovidModel):
    ydf = model.solution_sum(['seir', 'age']).stack(level='age')
    dfs_by_group = []
    for i, group in enumerate(model.attr['age']):
        dfs_by_group.append(ydf.xs(group, level='age').rename(columns={var: var + str(i+1) for var in model.attr['seir']}))
    df = pd.concat(dfs_by_group, axis=1)

    params_df = model.params_as_df

    totals = model.solution_sum('seir')
    by_age = model.solution_sum(['seir', 'age'])
    df['Iht'] = totals['Ih']
    df['Dt'] = totals['D']
    df['Rt'] = totals['R']
    df['Itotal'] = totals['I'] + totals['A']
    df['Etotal'] = totals['E']
    df['Einc'] = df['Etotal'] / model.specifications.model_params['alpha']
    for i, age in enumerate(model.attr['age']):
        df[f'Vt{i+1}'] = (model.solution_ydf[('S', age, 'vacc')] + model.solution_ydf[('R', age, 'vacc')]) * params_df.xs((age, 'vacc'), level=('age', 'vacc'))['vacc_eff']
        df[f'immune{i+1}'] = by_age[('R', age)] + model.solution_ydf[('S', age, 'vacc')] * params_df.xs((age, 'vacc'), level=('age', 'vacc'))['vacc_eff']
    df['Vt'] = sum(df[f'Vt{i+1}'] for i in range(4))
    df['immune'] = sum(df[f'immune{i+1}'] for i in range(4))
    df['date'] = model.daterange
    df['Ilag'] = totals['I'].shift(3)
    df['Re'] = model.re_estimates
    df['prev'] = 100000.0 * df['Itotal'] / model.specifications.model_params['total_pop']
    df['oneinX'] = model.specifications.model_params['total_pop'] / df['Itotal']
    df['Exposed'] = 100.0 * df['Einc'].cumsum()

    df.index.names = ['t']
    return df


def build_tc_df(model: CovidModel):
    return pd.DataFrame.from_dict({'time': model.tslices[:-1]
                                , 'tc_pb': model.efs
                                , 'tc': model.obs_ef_by_slice})


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
    print('Scenario tags: ', model.specifications.tags)
    model.solve_seir()
    model.write_to_db(engine, new_spec=True)
    if legacy_output_dict is not None:
        legacy_output_dict[tags_to_scen_label(model.specifications.tags)] = build_legacy_output_df(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--current_fit_id", type=int, help="The current fit ID (run today)")
    parser.add_argument("-pf", "--prior_fit_id", type=int, help="The prior fit ID (run last week)")
    parser.add_argument("-d", "--days", type=int, help="Number of days to include in these scenarios, starting from Jan 24, 2020")
    parser.add_argument("-tcs", "--tc_shifts", nargs='+', type=float, help="Upcoming shifts in TC to simulate (-0.05 represents a 5% reduction in TC)")
    parser.add_argument("-p", "--params", type=str, help="the path to the params file to use for fitting; default to 'input/params.json'")
    # parser.add_argument("-pvs", "--primary_vaccine_scen", choices=['high', 'low'], type=float, help="The name of the vaccine scenario to be used for the default model scenario.")
    run_params = parser.parse_args()

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
    specs = CovidModelSpecifications.from_db(engine, current_fit_id, new_end_date=model.end_date)
    specs.base_spec_id = current_fit_id
    specs.tags = {'batch': batch}
    vacc_proj_dict = json.load(open('input/vacc_proj_params.json'))
    for vacc_scen in vacc_scens:
        print(f'Prepping model with vaccine projection scenario "{vacc_scen}"...')
        specs.set_vacc_proj(vacc_proj_dict[vacc_scen])
        model.prep(specs=specs)
        print(f'Running scenarios...')
        model.specifications.tags.update({'run_type': 'Vaccination Scenario', 'vacc_cap': vacc_scen, 'tc_shift': 'no shift', 'tc_shift_date': 'no_shift'})
        run_model(model, engine, legacy_output_dict=legacy_outputs)
        if vacc_scen == primary_vacc_scen:
            model.specifications.tags.update({'run_type': 'Current', 'vacc_cap': vacc_scen})
            run_model(model, engine, legacy_output_dict=legacy_outputs)

        base_tslices, base_tc = (model.specifications.tslices, model.specifications.tc)
        for tcs in tc_shifts:
            tc_shift_t = (tc_shift_date - model.start_date).days
            model.apply_tc(tc=base_tc + list(np.linspace(base_tc[-1], base_tc[-1] + tcs, tc_shift_days)),
                           tslices=base_tslices +  list(range(tc_shift_t, tc_shift_t + tc_shift_days)))
            model.specifications.tags.update({'run_type': 'TC Shift Projection', 'tc_shift': f'{int(100 * tcs)}%', 'tc_shift_date': tc_shift_date.strftime("%b %#d")})
            run_model(model, engine, legacy_output_dict=legacy_outputs)

    df = pd.concat(legacy_outputs)
    df.index.names = ['scenario', 'time']
    df.to_csv('output/allscenarios.csv')

    exit()
    #
    #     # vacc cap scenarios
    #         for vacc_scen in models_by_vacc_scen.keys():
    #             tags = {'run_type': 'Vaccination Scenario', 'batch': batch, 'vacc_cap': vacc_scen}
    #             run_model(models_by_vacc_scen[vacc_scen], current_fit_id, fit_tags=tags)
    #             # run_model(models_with_increased_under20_inf_prob[vacc_scen], current_fit_id, fit_tags={**tags, **{'tc_shift': 'increased under-18 transm.'}})
    #
    #         # tc shift scenarios
    #         for tcs in tc_shifts:
    #             for tcsd in tc_shift_dates:
    #                 for vacc_scen in models_by_vacc_scen.keys():
    #                     for tc_shift_days in tc_shift_dayss:
    #                         tcsd_label = tcsd.strftime("%b %#d")
    #                         if tc_shift_days is not None:
    #                             tcsd_label += f' - {(tcsd + dt.timedelta(days=tc_shift_days)).strftime("%b %#d")}'
    #                         tags = {'run_type': 'TC Shift Projection', 'batch': batch, 'tc_shift': f'{int(100 * tcs)}%',
    #                                 'tc_shift_date': tcsd_label, 'vacc_cap': vacc_scen}
    #                         run_model(models_by_vacc_scen[vacc_scen], current_fit_id, tc_shift=tcs, tc_shift_date=tcsd,
    #                                   fit_tags=tags, tc_shift_length=tc_shift_length, tc_shift_days=tc_shift_days)
    #                         # run_model(models_with_increased_under20_inf_prob[vacc_scen], current_fit_id, tc_shift=tcs, tc_shift_date=tcsd, tc_shift_length=tc_shift_length, fit_tags={**tags, **{'tc_shift': tags['tc_shift'] + '; increased under-18 transm.'}})
    #
    # exit()
    #
    #
    #
    #
    #
    # vacc_proj_dict = json.load(open('input/vacc_proj_params.json'))
    # models_by_vacc_scen = {}
    # for vacc_scen, proj_params in vacc_proj_dict.items():
    #     vacc_proj_params = vacc_proj_dict[vacc_scen]
    #     print(f'Building {vacc_scen} projection...')
    #     models_by_vacc_scen[vacc_scen] = CovidModel(tslices=[0, tmax], engine=engine)
    #     models_by_vacc_scen[vacc_scen].set_ef_from_db(current_fit_id)
    #     models_by_vacc_scen[vacc_scen].prep(params=params_fname, vacc_proj_params=vacc_proj_params)
    #
    # def run_model(model, fit_id, fit_tags=None, tc_shift=None, tc_shift_date=None, tc_shift_length=None, tc_shift_days=None):
    #     print('Scenario tags: ', fit_tags)
    #     model.set_ef_from_db(fit_id)
    #     if tc_shift_days is not None and tc_shift_date is not None:
    #         model.fixed_tslices[-2] = min(model.fixed_tslices[-2], (tc_shift_date - model.start_date).days - 1)
    #     current_ef = model.tc[-1]
    #     if tc_shift is not None:
    #         if tc_shift_days is None:
    #             model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days, current_ef + tc_shift)
    #         else:
    #             for i, tc_shift_for_this_day in enumerate(np.linspace(0, tc_shift, tc_shift_days)):
    #                 model.add_tslice((tc_shift_date - dt.datetime(2020, 1, 24)).days + i, current_ef + tc_shift_for_this_day)
    #
    #         if tc_shift_length is not None:
    #             model.add_tslice(((tc_shift_date + dt.timedelta(days=tc_shift_length)) - dt.datetime(2020, 1, 24)).days, current_ef)
    #
    #     model.solve_seir()
    #     model.write_to_db(tags=fit_tags, new_fit=True)
    #     legacy_outputs[tags_to_scen_label(fit_tags)] = build_legacy_output_df(model)
    #     return model
    #
    # # current fit
    # tags = {'run_type': 'Current', 'batch': batch}
    # run_model(models_by_vacc_scen[primary_vacc_scen], current_fit_id, fit_tags=tags)
    # # output this one to it's own file
    # build_legacy_output_df(models_by_vacc_scen[primary_vacc_scen]).to_csv('output/out2.csv')
    # # and output the TCs to their own file
    # build_tc_df(models_by_vacc_scen[primary_vacc_scen]).to_csv('output/tc_over_time.csv', index=False)
    #
    # # prior fit
    # # tags = {'run_type': 'Prior', 'batch': batch}
    # # prior_fit_model = CovidModel(tslices=[0, tmax], engine=engine)
    #
    # # prior_fit_model.prep(params=params_fname, vacc_proj_params=vacc_proj_dict[primary_vacc_scen])
    # # run_model(prior_fit_model, prior_fit_id, fit_tags=tags)
    #
    # # vacc cap scenarios
    # for vacc_scen in models_by_vacc_scen.keys():
    #     tags = {'run_type': 'Vaccination Scenario', 'batch': batch, 'vacc_cap': vacc_scen}
    #     run_model(models_by_vacc_scen[vacc_scen], current_fit_id, fit_tags=tags)
    #     # run_model(models_with_increased_under20_inf_prob[vacc_scen], current_fit_id, fit_tags={**tags, **{'tc_shift': 'increased under-18 transm.'}})
    #
    # # tc shift scenarios
    # for tcs in tc_shifts:
    #     for tcsd in tc_shift_dates:
    #         for vacc_scen in models_by_vacc_scen.keys():
    #             for tc_shift_days in tc_shift_dayss:
    #                 tcsd_label = tcsd.strftime("%b %#d")
    #                 if tc_shift_days is not None:
    #                     tcsd_label += f' - {(tcsd + dt.timedelta(days=tc_shift_days)).strftime("%b %#d")}'
    #                 tags = {'run_type': 'TC Shift Projection', 'batch': batch, 'tc_shift': f'{int(100 * tcs)}%', 'tc_shift_date': tcsd_label, 'vacc_cap': vacc_scen}
    #                 run_model(models_by_vacc_scen[vacc_scen], current_fit_id, tc_shift=tcs, tc_shift_date=tcsd, fit_tags=tags, tc_shift_length=tc_shift_length, tc_shift_days=tc_shift_days)
    #                 # run_model(models_with_increased_under20_inf_prob[vacc_scen], current_fit_id, tc_shift=tcs, tc_shift_date=tcsd, tc_shift_length=tc_shift_length, fit_tags={**tags, **{'tc_shift': tags['tc_shift'] + '; increased under-18 transm.'}})
    #
    # # for vacc_scen in models_by_vacc_scen.keys():
    # #     model = CovidModel(params='input/params.json', tslices=[0, tmax], engine=engine)
    # #     model.gparams['rel_inf_prob'] = {'tslices': [577, 647], 'value': {'0-19': [1.0, 2.0, 2.0], '20-39': [1.0, 1.0, 1.0], '40-64': [1.0, 1.0, 1.0], '65+': [1.0, 1.0, 1.0]}}
    # #     model.prep(vacc_proj_scen=vacc_scen)
    # #     run_model(model, current_fit_id, tc_shift=0.05, tc_shift_date=dt.date(2021, 8, 27), tc_shift_length=70, fit_tags={}),
    #
    # # custom scenarios
    # # for vacc_scen in models_by_vacc_scen.keys():,
    # # for scen in models_for_custom_scenarios.keys():
    # #     for vacc_scen in models_for_custom_scenarios[scen].keys():
    # #         tags = {'run_type': 'Custom Projection', 'batch': batch, 'tc_shift': scen, 'vacc_cap': vacc_scen}
    # #         run_model(models_for_custom_scenarios[scen][vacc_scen], current_fit_id, tc_shift, )

    df = pd.concat(legacy_outputs)
    df.index.names = ['scenario', 'time']
    df.to_csv('output/allscenarios.csv')


if __name__ == '__main__':
    main()
    # engine = db_engine()
    #
    # vacc_proj_dict = json.load(open('input/vacc_proj_params.json'))
    # for vacc_scen, proj_params in vacc_proj_dict.items():
    #     model = CovidModel()
    #     specs = CovidModelSpecifications.from_db(engine, 121, new_end_date=model.end_date)
    #
    #
    #
    # model = CovidModel()
    # model.prep(engine=engine, specs=121)
    # model.solve_seir()
    #
    #
    #
    # model.write_to_db(tags=fit_tags, new_fit=True)
    # legacy_outputs[tags_to_scen_label(fit_tags)] = build_legacy_output_df(model)

