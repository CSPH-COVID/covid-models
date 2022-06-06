### Python Standard Library ###
import copy
import os
import datetime as dt
import json
import logging
import numpy as np
### Third Party Imports ###
from matplotlib import pyplot as plt
### Local Imports ###
from covid_model import CovidModel
from covid_model.utils import IndentLogger, setup, get_filepath_prefix
logger = IndentLogger(logging.getLogger(''), {})
from covid_model.db import db_engine
from covid_model.analysis.charts import plot_transmission_control
from covid_model.runnable_functions import do_create_report, do_build_legacy_output_df

def main():
    ####################################################################################################################
    # Set Up Arguments for Running
    outdir = setup(os.path.basename(__file__), 'info')

    model_args = {'base_spec_id': 2665, 'end_date': "2022-11-01", 'params_defs': 'covid_model/analysis/20220525_model_simplification/params_nopriorinf.json',  'max_step_size': 1.0}
    logging.info(json.dumps({"model_args": model_args}, default=str))

    ####################################################################################################################
    # Run

    # fit a statewide model up to present day to act as a baseline
    logging.info('Loading Model')

    model = CovidModel(**model_args)
    model.prep()
    model.solve_seir()
    do_build_legacy_output_df(model).to_csv(get_filepath_prefix(outdir) + "out2.csv")

    do_create_report(model, outdir)

    logger.info(f'{str(model.tags)}: Running forward sim')
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(211)
    hosps_df = model.modeled_vs_actual_hosps().reset_index('region').drop(columns='region')
    hosps_df.plot(ax=ax)
    ax = fig.add_subplot(212)
    plot_transmission_control(model, ax=ax)
    plt.savefig(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_fit.png')
    plt.close()
    hosps_df.to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_fit.csv')
    json.dump(dict(dict(zip([0] + model.tc_tslices, model.tc))), open(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_tc.json', 'w'))

    logger.info(f'{str(model.tags)}: Uploading final results')
    engine = db_engine()
    model.write_specs_to_db(engine)
    model.write_results_to_db(engine)
    logger.info(f'{str(model.tags)}: spec_id: {model.spec_id}')
    model.solution_sum(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir) + "states_seir_variant_immun.csv")


if __name__ == "__main__":
    main()