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
from covid_model.runnable_functions import do_single_fit
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
        #'tc_window_batch_size': 3,
        #'tc_batch_increment': 1,
        'last_tc_window_min_size': 14,
        #'write_results': True, TODO: doesn't currently work with multiprocessing options. fix this. (still writes, but can't specify arguments)
        'outdir': outdir
    }
    model_args = {
        'params_defs': 'covid_model/analysis/20220616_mobility_update/params.json',
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/20220718_vacc_proj_params.json',
        #'attrs': OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
        #                      'age': ['0-19', '20-39', '40-64', '65+'],
        #                      'vacc': ['none', 'shot1', 'shot2', 'shot3'],
        #                      'variant': ['none', 'wildtype'],
        #                      'immun': ['none', 'weak', 'strong'],
        #                      'region': ['cent', 'cm', 'met', 'ms', 'ne', 'nw', 'slv', 'sc', 'sec', 'sw', 'wcp']}),
        #'regions': ['met', 'ms', 'sc', 'ne', 'nw'],
        'regions': ['met'],
        #'regions': ['cent', 'cm', 'met', 'ms', 'ne', 'nw', 'slv', 'sc', 'sec', 'sw', 'wcp'],
        'mobility_mode': 'population_attached',
        #'mobility_mode': None,
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2021-11-09', "%Y-%m-%d").date(),
        #'max_step_size': np.inf
        'max_step_size': 1.0,
        'ode_method': 'RK23'
    }
    #base_model_args = {'base_spec_id': 2710, 'params_defs': json.load(open('covid_model/analysis/20220606_GovBriefing/params_no_ba45_immuneescape.json'))}
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"model_args": model_args}, default=str))

    ####################################################################################################################
    # Run

    logging.info('Building Scenarios')
    scenario_args_list = []


    # run the scenarios
    logging.info('Fitting Model')

    model = do_single_fit(**fit_args, tc_window_batch_size=2, tc_batch_increment=1, write_results=False, **model_args)
    model = do_single_fit(**fit_args, tc_window_batch_size=4, tc_batch_increment=2, write_results=True, tc_0=None, base_model = model)

    #logging.info('Projecting')

    #logging.info('')

    #model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_seir_variant_immun_total_all_at_once_projection.csv')
    #model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_full_projection.csv')

    #logging.info(f'{str(model.tags)}: Running forward sim')
    #fig = plt.figure(figsize=(10, 10), dpi=300)
    #ax = fig.add_subplot(211)
    #hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
    #hosps_df.plot(ax=ax)
    #ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2022-09-15', "%Y-%m-%d").date())
    #ax = fig.add_subplot(212)
    #plot_transmission_control(model, ax=ax)
    #ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(), dt.datetime.strptime('2022-09-15', "%Y-%m-%d").date())
    #plt.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast.png')
    ##plt.close()
    #hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_forecast.csv')
    #json.dump(model.tc, open(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast_tc.json', 'w'))








if __name__ == "__main__":
    main()