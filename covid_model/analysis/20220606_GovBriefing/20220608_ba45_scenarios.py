### Python Standard Library ###
import copy
import os
import datetime as dt
import json
import logging
import numpy as np
### Third Party Imports ###
from collections import OrderedDict
from matplotlib import pyplot as plt
### Local Imports ###
from covid_model import CovidModel
from covid_model.runnable_functions import do_create_report, do_fit_scenarios
from covid_model.utils import setup, get_filepath_prefix
from covid_model.analysis.charts import plot_transmission_control
from covid_model.db import db_engine


def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    outdir = setup(os.path.basename(__file__), 'info')

    fit_args = {
        'batch_size': 5,
        'increment_size': 2,
        'tc_min': 0.0,
        'tc_max': 0.999,
        'forward_sim_each_batch': True,
        'look_back': None,
        #'refit_from_date': '2022-03-01',
        'use_hosps_end_date': True,
        'window_size': 14,
        'last_window_min_size': 14,
        'write_batch_output': False,
        'outdir': outdir
    }
    base_model_args = {
        'params_defs': json.load(open('covid_model/analysis/20220606_GovBriefing/params_no_ba45_immuneescape.json')),
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',
        'regions': ['co'],
        'tc': [0.75, 0.75],
        'tc_tslices': [14],
        'mobility_mode': None,
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-06-07', "%Y-%m-%d").date(),
        #'max_step_size': np.inf
        'max_step_size': 1.0
    }
    multiprocess = 6
    base_model_args = {'base_spec_id': 2710, 'params_defs': json.load(open('covid_model/analysis/20220606_GovBriefing/params_no_ba45_immuneescape.json'))}
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"base_model_args": base_model_args}, default=str))

    ####################################################################################################################
    # Run

    scenario_args_list = []
    #for beta_mult in [0.9, 0.95, 0.99, 1.01, 1.05, 1.1]:
    for beta_mult in [0.95, 1.00, 1.05]:
        for (weak_escape, strong_escape) in [(0.75, 0.1), (0.8, 0.2)]:
            weak_param = [{"param": "immune_escape", "from_attrs": {"immun": "weak", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": weak_escape},  "desc": "weak"}]
            strong_param = [{"param": "immune_escape", "from_attrs": {"immun": "strong", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": strong_escape},  "desc": "strong"}]
            beta_param_adjustment = [{"param": "betta",  "attrs": {"variant": "ba45"}, "mults":  {"2020-01-01": beta_mult}, "desc": "sensitivity"}]
            scenario_args_list.append({'params_defs': base_model_args['params_defs'] + beta_param_adjustment + weak_param + strong_param,
                                       'tags': {'beta_mult': beta_mult, 'ba45_escape_weak': weak_escape, 'ba45_escape_strong': strong_escape}})

    # run the scenarios
    models = do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess=multiprocess)

    for model in models:
        logging.info('Fitting')
        model.solution_sum(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_seir_variant_immun_total_all_at_once.csv')
        model.solution_sum().unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_full.csv')

        model.end_date = '2022-09-15'
        model.update(db_engine())
        do_create_report(model, outdir, prep_model=True, solve_model=True)

        model.solution_sum(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_seir_variant_immun_total_all_at_once_forecast.csv')
        model.solution_sum().unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_full_forecast.csv')

        logging.info(f'{str(model.tags)}: Running forward sim')
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(211)
        hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
        hosps_df.plot(ax=ax)
        ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2022-09-15', "%Y-%m-%d").date())
        ax = fig.add_subplot(212)
        plot_transmission_control(model, ax=ax)
        ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2022-09-15', "%Y-%m-%d").date())
        plt.savefig(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_forecast.png')
        plt.close()
        hosps_df.to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_forecast.csv')
        json.dump(dict(dict(zip([0] + model.tc_tslices, model.tc))), open(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_forecast_tc.json', 'w'))


if __name__ == "__main__":
    main()