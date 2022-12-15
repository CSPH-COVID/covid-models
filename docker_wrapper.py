#=======================================================================================================================
# docker_wrapper.py
# Written by: Andrew Hill
# Last Modified: 11/16/2022
# Description:
#   This file serves as an entry point for a Docker container to process a single region of the CSTE Rocky Mountain West
#   model.
#   This file assumes that the region to run is defined as the environment variable "RMW_REGION".
#   In the future this file may also support passing in separate parameter files for each region.
#=======================================================================================================================
import datetime
import os
import json
import base64
import sys
import logging
from matplotlib import pyplot as plt
from covid_model.rmw_model import RMWCovidModel
from covid_model.runnable_functions import do_single_fit, do_create_report
from covid_model.utils import setup, get_filepath_prefix
from covid_model.analysis.charts import plot_transmission_control

TESTING = True

if __name__ == "__main__":
    # ENVIRONMENT VARIABLE SETUP
    # The BATCH_TASK_INDEX variable is passed in by Batch, and is the unique index of the Task within the group.
    # We can use this to index into the list of regions, to determine which region we should fit.
    BATCH_TASK_INDEX = int(os.environ["BATCH_TASK_INDEX"])
    # The model references the 'gcp_project' environment variable later, so we hard-code it here.
    os.environ["gcp_project"] = "co-covid-models"

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

    if TESTING:
        with open("sample_config.json","r") as f:
            args = json.load(f)
    else:
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

    # OUTPUT SETUP
    # The region handled by this Task/instance is just the BATCH_TASK_INDEX-th element of the args["regions"] list.
    instance_region = args["regions"][BATCH_TASK_INDEX]
    outdir = setup(name=instance_region,log_level="info")

    # MODEL SETUP
    # Retrieve the parameters for the model.
    # For now we expect that all parameters for all regions exist in the same file.
    # TODO: Change this to support separate parameter files per-region.
    model_args = {
        'params_defs': 'covid_model/input/rmw_temp_params.json',
        'region_defs': 'covid_model/input/rmw_region_definitions.json',
        'vacc_proj_params': 'covid_model/input/rmw_vacc_proj_params.json',
        'start_date': args["start_date"],
        'end_date': args["end_date"],
        'regions': [instance_region]
    }

    fit_args = {'outdir': outdir,
                'fit_end_date': args["fit_end_date"],
                'model_class': RMWCovidModel}

    # MODEL FITTING
    # This code is mostly just copied from the Jupyter notebooks we use, but in the future we can make this
    # a more general wrapper for doing model fitting and generating plots.
    model = do_single_fit(**model_args,**fit_args)
    #model = RMWCovidModel(base_spec_id=4578)
    #model.update_tc({model.date_to_t(args["fit_end_date"]): {instance_region: 0.5}}, replace = False)
    model.solve_seir()
    # MODEL OUTPUTS
    logging.info('Projecting')
    do_create_report(model, outdir=outdir, prep_model=False, solve_model=False, immun_variants=args["report_variants"],
                     from_date=args["report_start_date"])
    model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(
        get_filepath_prefix(outdir, tags=model.tags) + 'reg_seir_variant_immun_total_all_at_once_projection.csv')
    model.solution_sum_df().unstack().to_csv(
        get_filepath_prefix(outdir, tags=model.tags) + 'reg_full_projection.csv')

    logging.info(f'{str(model.tags)}: Running forward sim')

    # MODEL PLOTS
    xmin = datetime.datetime.strptime(args["report_start_date"],"%Y-%m-%d").date()
    xmax = datetime.datetime.strptime(args["end_date"],"%Y-%m-%d").date()
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(211)
    hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
    hosps_df.plot(ax=ax)
    ax.set_xlim(xmin, xmax)
    ax = fig.add_subplot(212)
    plot_transmission_control(model, ax=ax)
    ax.set_xlim(xmin, xmax)
    plt.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast.png')
    plt.close()
    hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_forecast.csv')
    json.dump(model.tc, open(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast_tc.json', 'w'))

    logging.info("Task finished.")