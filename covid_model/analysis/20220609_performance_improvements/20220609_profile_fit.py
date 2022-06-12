### Python Standard Library ###
import copy
import os
import datetime as dt
import json
import logging
import numpy as np
from time import perf_counter
### Third Party Imports ###
from collections import OrderedDict
from matplotlib import pyplot as plt
#from line_profiler_pycharm import profile
### Local Imports ###
from covid_model import CovidModel
from covid_model.runnable_functions import do_single_fit, do_create_report
from covid_model.utils import setup, get_filepath_prefix
from covid_model.analysis.charts import plot_transmission_control
from covid_model.db import db_engine


#@profile
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
        'write_results': False,
        'loss_projection_days': 0,
        'outdir': outdir
    }
    model_args = {
        'params_defs': json.load(open('covid_model/analysis/20220609_performance_improvements/profile_params.json')),
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',
        'regions': ['co'],
        'mobility_mode': None,
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2020-08-01', "%Y-%m-%d").date(),
        # 'max_step_size': np.inf
        'max_step_size': 1.0
    }
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"model_args": model_args}, default=str))

    ####################################################################################################################

    #model = CovidModel(**model_args)
    #t0 = perf_counter()
    #model.prep()
    #model.update_tc({0: 0.75})
    #logging.info(f'{str(model.tags)} Model prepped for fitting in {perf_counter() - t0} seconds.')
    #t1 = perf_counter()
    #for i in range(100):
    #    model.solve_seir()
    #logging.info(f'{str(model.tags)} Model fit 100 times in {perf_counter() - t1} seconds.')


    logging.info('Fitting')
    model = do_single_fit(**fit_args, **model_args, tags={'priorinf': False})
    model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + "states_seir_variant_immun_total_all_at_once.csv")
    model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir) + "states_full.csv")
    logging.debug(json.dumps({"serialized model": model.to_json_string()}, default=str))

    return

    model.end_date = '2022-09-15'
    model.update(db_engine())
    do_create_report(model, outdir, prep_model=True, solve_model=True)

    logging.info('Fitting')
    model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_seir_variant_immun_total_all_at_once.csv')
    model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_full.csv')

    model.end_date = '2022-09-15'
    model.update(db_engine())
    do_create_report(model, outdir, prep_model=True, solve_model=True)

    model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_seir_variant_immun_total_all_at_once_forecast.csv')
    model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_states_full_forecast.csv')

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
    json.dump(dict(dict(zip(model.tc_tslices, model.tc))), open(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_forecast_tc.json', 'w'))


if __name__ == "__main__":
    main()