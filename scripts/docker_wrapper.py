#=======================================================================================================================
# docker_wrapper.py
# Written by: Andrew Hill
# Last Modified: 1/27/2023
# Description:
#   This file serves as an entry point for a Docker container to process a single region of the CSTE Rocky Mountain West
#   model.
#   This file is intended to be run using Google Batch, the region used for model run is chosen based on the value of
#   BATCH_TASK_INDEX, an environment variable defined by the Google Batch system.
#=======================================================================================================================
import datetime
import os
import json
import base64
import sys
import logging
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from covid_model.rmw_model import RMWCovidModel
from covid_model.runnable_functions import do_single_fit, do_create_report, do_fit_scenarios, do_create_multiple_reports
from covid_model.utils import setup, get_filepath_prefix
from covid_model.analysis.charts import plot_transmission_control


def wrapper_run(args: dict):
    # ENVIRONMENT VARIABLE SETUP
    # The BATCH_TASK_INDEX variable is passed in by Google Batch, and is the unique index of the Task within the group.
    # We can use this to index into the list of regions and determine which region we should fit.
    BATCH_TASK_INDEX = int(os.environ["BATCH_TASK_INDEX"])
    # Because we are running multiple instances of the model concurrently, we need a way to prevent race-conditions
    # from occurring if multiple instances are concurrently trying to obtain spec_ids. The script or workflow which
    # triggered this run should pass us the START_SPEC_ID environment variable, which is the first unused spec_id
    # within the database. Using this, we can obtain unique spec_ids for each model/scenario, since we know:
    # 1. Which region we are running (BATCH_TASK_INDEX)
    # 2. How many spec_ids will be consumed per concurrent model run (n_scenarios + 1)
    START_SPEC_ID = int(os.environ["SPEC_ID"])
    # The model references the 'gcp_project' environment variable later, so we hard-code it here.
    os.environ["gcp_project"] = "co-covid-models"
    # OUTPUT SETUP
    # The region handled by this Task/instance is just the BATCH_TASK_INDEX-th element of the args["regions"] list.
    instance_region = args["regions"][BATCH_TASK_INDEX]
    outdir = setup(name=instance_region, log_level="info")
    # Set up the scenario fit to start the day after the base_model fit ends.
    scenario_fit_start_date = (pd.to_datetime(args["base_model_fit_end_date"]) + pd.Timedelta(1, unit="D")) \
        .strftime("%Y-%m-%d")

    # MODEL SETUP
    # Retrieve the parameters for the model.
    # For now we expect that all parameters for all regions exist in the same file.
    # TODO: Change this to support separate parameter files per-region.
    base_model_args = {
        'params_defs': 'covid_model/input/rmw_temp_params.json',
        # 'region_defs': 'covid_model/input/rmw_region_definitions.json',
        'vacc_proj_params': 'covid_model/input/rmw_vacc_proj_params.json',
        'start_date': args["start_date"],
        'end_date': args["end_date"],
        'fit_end_date': args["base_model_fit_end_date"],
        'regions': [instance_region],
        'tags': {"region": instance_region},
        'outdir': outdir,
        'pickle_matrices': False
    }

    scenario_fit_args = {
        'outdir': outdir,
        'fit_start_date': scenario_fit_start_date,
        'fit_end_date': args["scenarios_fit_end_date"],
        'model_class': RMWCovidModel,
        'write_results': False,
        'pickle_matrices': False,
        'pre_solve_model': True
    }
    # Set up the arguments for the scenario fits.
    vacc_eff_lt5 = 0.5
    scenario_model_args = []
    scenario_params = json.load(open(base_model_args["params_defs"]))
    for (weak_escape, strong_escape) in [(0.6, 0.15), (0.66, 0.165), (0.72, 0.18), (0.54, 0.135)]:
        weak_param = [{"param": "immune_escape",
                       "from_attrs": {"immun": "weak",
                                      "variant": ["none", "wildtype", "alpha", "delta",
                                                  "omicron", "ba2", "ba2121", "ba45","bq"]},
                       "to_attrs": {"variant": ["xbb"]},
                       "vals": {"2020-01-01": weak_escape},
                       "desc": "emerging variants immune escape value, weak immunity"}]
        strong_param = [{"param": "immune_escape",
                         "from_attrs": {"immun": "strong",
                                        "variant": ["none", "wildtype", "alpha", "delta",
                                                    "omicron", "ba2", "ba2121", "ba45", "bq"]},
                         "to_attrs": {"variant": ["xbb"]},
                         "vals": {"2020-01-01": strong_escape},
                         "desc": "emerging variants immune escape value, strong immunity"}]
        # lt5_vacc_adjust = [{"param": "immunity",
        #                     "attrs": {'age': '0-19', 'vacc': 'shot1'},
        #                     "mults": {"2020-01-01": 1,
        #                               "2022-06-24": 0.99 + 0.01 * vacc_eff_lt5,
        #                               "2022-06-30": 0.98 + 0.02 * vacc_eff_lt5,
        #                               "2022-07-08": 0.97 + 0.03 * vacc_eff_lt5,
        #                               "2022-07-19": 0.96 + 0.04 * vacc_eff_lt5,
        #                               "2022-07-29": 0.95 + 0.05 * vacc_eff_lt5,
        #                               "2022-08-11": 0.94 + 0.06 * vacc_eff_lt5,
        #                               "2022-08-30": 0.93 + 0.07 * vacc_eff_lt5,
        #                               "2022-09-26": 0.92 + 0.08 * vacc_eff_lt5,
        #                               "2022-10-26": 0.91 + 0.09 * vacc_eff_lt5, },
        #                     "desc": "weighted average using share of 0-19 getting shot1 who are under 5"}]
        scenario_model_args.append({'params_defs': scenario_params + weak_param + strong_param,
                                    'tags': {'emv_escape_weak': weak_escape,
                                             'emv_escape_strong': strong_escape}})
    # SET SPEC IDs
    # Number of fits is the number of scenarios plus the base model fit
    n_fits = len(scenario_model_args) + 1
    spec_ids = [START_SPEC_ID + (BATCH_TASK_INDEX * n_fits) + i for i in range(n_fits)]
    base_model_args["spec_id"] = spec_ids[0]
    for scen, spec_id in zip(scenario_model_args, spec_ids[1:]):
        scen["base_spec_id"] = spec_ids[0]
        scen["spec_id"] = spec_id
    # MODEL FITTING
    # This code is mostly just copied from the Jupyter notebooks we use, but in the future we can make this
    # a more general wrapper for doing model fitting and generating plots.
    base_model = do_single_fit(**base_model_args)
    #base_model = RMWCovidModel(base_spec_id=4864)
    #base_model.prep()
    base_model.solve_seir()
    with open(get_filepath_prefix(outdir, tags=base_model.tags) + f"model_solutionydf.pkl", "wb") as f:
        pickle.dump(base_model.solution_ydf, f)
    #base_model.solve_seir()

    # MODEL OUTPUTS
    logging.info('Projecting')
    do_create_report(base_model, outdir=outdir, prep_model=False, solve_model=False,
                     immun_variants=args["report_variants"],
                     from_date=args["report_start_date"])
    base_model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(
        get_filepath_prefix(outdir, tags=base_model.tags) + 'seir_variant_immun_total_all_at_once_projection.csv')
    base_model.solution_sum_df().unstack().to_csv(
        get_filepath_prefix(outdir, tags=base_model.tags) + 'full_projection.csv')

    # SCENARIO FITTING
    logging.info(f"{str(base_model.tags)}: Running scenarios")
    models = do_fit_scenarios(base_model_args=base_model_args, scenario_args_list=scenario_model_args,
                              fit_args=scenario_fit_args)
    for model in models:
        xmin = datetime.datetime.strptime(args["report_start_date"], "%Y-%m-%d").date()
        xmax = datetime.datetime.strptime(args["end_date"], "%Y-%m-%d").date()
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(211)
        hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
        hosps_df.plot(ax=ax)
        ax.set_xlim(xmin, xmax)
        ax = fig.add_subplot(212)
        plot_transmission_control(model, ax=ax)
        ax.set_xlim(xmin, xmax)
        plt.savefig(get_filepath_prefix(outdir, tags=model.tags) + '_model_forecast.png')
        plt.close()
        hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_forecast.csv')
        json.dump(model.tc, open(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast_tc.json', 'w'))

    logging.info("Task finished.")


if __name__ == "__main__":
    # INPUT PARAMETERS
    # The input parameters to the container are passed in as a base64 encoded JSON string. To retrieve them in a usable
    # format, we first need to decode the base64 string into a UTF-8 JSON string, then parse the JSON string to retrieve
    # a Python dictionary which we can use the arguments to the model.
    # NOTE: All batch instances receive the same input arguments. The BATCH_TASK_INDEX variable can be used to index
    # into instance-specific arguments (like region).
    # The minimum viable JSON parameter object is:
    # {
    #   "regions" : [...],
    #   "start_date": <start_date>,
    #   "end_date": <end_date>,
    #   "fit_end_date": <fit_end_date>,
    #   "report_start_date": <report_start_date>,
    #   "report_variants": [...]
    # }

    # If any of these fields are not defined in the input JSON, the program will fail as it expects them to exist.
    if len(sys.argv) < 2:
        print("Error: Missing input arguments.")
        sys.exit(1)
    # Retrieve B64-encoded string.
    b64_json_str = sys.argv[1]
    # Decode the string
    json_str = base64.b64decode(b64_json_str, validate=True)
    # Load the JSON string
    args = json.loads(json_str)
    # Run the wrapper function with these input arguments.
    wrapper_run(args)
