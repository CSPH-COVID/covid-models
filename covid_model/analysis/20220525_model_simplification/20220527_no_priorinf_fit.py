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
from covid_model.runnable_functions import do_single_fit
from covid_model.utils import setup, get_filepath_prefix
from covid_model.analysis.charts import plot_transmission_control
from covid_model.db import db_engine


def remove_variants(model, y_dict, remove_variants):
    new_variant = [var for var in model.attrs['variant'] if var not in remove_variants][0]
    for variant in remove_variants:
        cmpts_to_remove = model.get_cmpts_matching_attrs({'variant': variant})
        for cmpt in cmpts_to_remove:
            new_cmpt = tuple(cmpt[:4] + tuple([new_variant]) + cmpt[5:])
            y_dict[new_cmpt] += y_dict[cmpt]
            del y_dict[cmpt]
    return y_dict


def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    outdir = setup(os.path.basename(__file__), 'info')

    fit_args = {
        'batch_size': 5,
        'increment_size': 2,
        'tc_min': -0.99,
        'tc_max': 0.999,
        'forward_sim_each_batch': True,
        'look_back': None,
        'use_hosps_end_date': True,
        'window_size': 14,
        'last_window_min_size': 7,  # make this smaller because we are refitting anyways
        'write_batch_output': False,
        'outdir': outdir
    }
    model_args = {
        'params_defs': 'covid_model/analysis/20220525_model_simplification/params_nopriorinf.json',
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',
        'regions': ['co'],
        'tc': [0.75, 0.75],
        'tc_tslices': [14],
        'mobility_mode': None,
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-05-27', "%Y-%m-%d").date(),
        #'max_step_size': np.inf
        'max_step_size': 1.0
    }
    #model_args = {'base_spec_id': 2647}
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"model_args": model_args}, default=str))

    ####################################################################################################################
    # Run

    # fit a statewide model up to present day to act as a baseline
    logging.info('Fitting')
    model = do_single_fit(**fit_args, **model_args, tags={'priorinf': False})
    model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + "states_seir_variant_immun_total_all_at_once.csv")
    model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir) + "states_full.csv")
    logging.debug(json.dumps({"serialized model": model.to_json_string()}, default=str))

    model.end_date = '2022-09-01'
    model.prep()  # needed
    model.solve_seir()
    logging.debug(json.dumps({"serialized model (new end date)": model.to_json_string()}, default=str))
    model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + "states_seir_variant_immun_total_all_at_once_forecast.csv")
    model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir) + "states_full_forecast.csv")

    logging.info(f'{str(model.tags)}: Running forward sim')
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(211)
    hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
    hosps_df.plot(ax=ax)
    ax = fig.add_subplot(212)
    plot_transmission_control(model, ax=ax)
    plt.savefig(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_forecast.png')
    plt.close()
    hosps_df.to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_forecast.csv')
    json.dump(dict(dict(zip([0] + model.tc_tslices, model.tc))), open(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_forecast_tc.json', 'w'))


if __name__ == "__main__":
    main()