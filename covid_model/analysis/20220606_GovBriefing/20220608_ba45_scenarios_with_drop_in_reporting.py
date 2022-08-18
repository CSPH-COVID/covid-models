""" Python Standard Library """
import copy
import os
import datetime as dt
import json
import logging
import numpy as np
""" Third Party Imports """
from collections import OrderedDict
from matplotlib import pyplot as plt
""" Local Imports """
from covid_model import CovidModel
from covid_model.runnable_functions import do_create_report, do_fit_scenarios, do_create_multiple_reports
from covid_model.utils import setup, get_filepath_prefix, db_engine
from covid_model.analysis.charts import plot_transmission_control


def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    outdir = setup(os.path.basename(__file__), 'info')

    fit_args = {
        'fit_start_date': None,
        'fit_end_date': None,
        'tc_min': 0.0,
        'tc_max': 0.999,
        'tc_window_size': 14,
        'tc_window_batch_size': 5,
        'tc_batch_increment': 2,
        'last_tc_window_min_size': 14,
        #'write_results': True, TODO: doesn't currently work with multiprocessing options. fix this. (still writes, but can't specify arguments)
        'outdir': outdir
    }
    base_model_args = {
        'params_defs': json.load(open('covid_model/analysis/20220606_GovBriefing/params_no_ba45_immuneescape_v3.json')),
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/20220718_vacc_proj_params.json',
        'regions': ['co'],
        'mobility_mode': None,
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-09-15', "%Y-%m-%d").date(),
        #'max_step_size': np.inf
        'max_step_size': 1.0
    }
    multiprocess = 9
    #base_model_args = {'base_spec_id': 2710, 'params_defs': json.load(open('covid_model/analysis/20220606_GovBriefing/params_no_ba45_immuneescape.json'))}
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"base_model_args": base_model_args}, default=str))

    ####################################################################################################################
    # Run

    logging.info('Building Scenarios')
    scenario_args_list = []
    #for beta_mult in [0.9, 0.95, 0.99, 1.01, 1.05, 1.1]:
    for beta_mult in [1.00, 0.95, 1.05]:
    #for beta_mult in [1.00]:
#        for for_covid_frac in [0.65, 0.5, 0.8]:
        for for_covid_frac in [0.8]:
            for (weak_escape, strong_escape) in [(0.75, 0.1), (0.8, 0.2)]:
                weak_param = [{"param": "immune_escape", "from_attrs": {"immun": "weak", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": weak_escape},  "desc": "weak"}]
                strong_param = [{"param": "immune_escape", "from_attrs": {"immun": "strong", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": strong_escape},  "desc": "strong"}]
                beta_param_adjustment = [{"param": "betta",  "attrs": {"variant": "ba45"}, "mults":  {"2020-01-01": beta_mult}, "desc": "sensitivity"}]
                scenario_args_list.append({'params_defs': base_model_args['params_defs'] + beta_param_adjustment + weak_param + strong_param,
                                           'tags': {'beta_mult': beta_mult, 'ba45_escape_weak': weak_escape, 'ba45_escape_strong': strong_escape, 'for_covid_frac': for_covid_frac},
                                           'hosp_reporting_frac': {"2020-01-01": 1, "2022-03-10": for_covid_frac}})

    # run the scenarios
    logging.info('Running Scenarios')
    models = do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess=multiprocess)

    #logging.info('Loading models from DB')
    #engine=db_engine()
    #models = [CovidModel(engine, base_spec_id=id) for id in [2772,2773,2774,2775,2776,2777,2778,2779,2780,2781,2782,2783,2784,2785,2786,2787,2788,2789]]
    #models = [CovidModel(engine, base_spec_id=id) for id in [2772,2773]]

    logging.info('Projecting')
    for model in models:
        logging.info('')
        #model.prep()  # don't think we need to prep anymore.
        model.solve_seir()

        model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_seir_variant_immun_total_all_at_once_projection.csv')
        model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_full_projection.csv')

        logging.info(f'{str(model.tags)}: Running forward sim')
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(211)
        hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
        hosps_df.plot(ax=ax)
        ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2022-09-15', "%Y-%m-%d").date())
        ax = fig.add_subplot(212)
        plot_transmission_control(model, ax=ax)
        ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2022-09-15', "%Y-%m-%d").date())
        plt.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast.png')
        plt.close()
        hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_forecast.csv')
        json.dump(model.tc, open(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast_tc.json', 'w'))

    logging.info('Running reports')
    do_create_multiple_reports(models, multiprocess=multiprocess, outdir=outdir, prep_model=False, solve_model=True, immun_variants=['ba2121', 'ba45'], from_date='2022-01-01')






if __name__ == "__main__":
    main()