""" Python Standard Library """
import os
import datetime as dt
import json
import logging
""" Third Party Imports """
from matplotlib import pyplot as plt
""" Local Imports """
from covid_model.runnable_functions import do_fit_scenarios, do_create_multiple_reports
from covid_model.utils import setup, get_filepath_prefix
from covid_model.analysis.charts import plot_transmission_control


"""
Welcome! This script is a companion to the Jupyter notebook 00000000_hello_world.ipynb.
That notebook is structured around the narrative of introducing someone to using our models for the first time.
This script is meant to be an example of how modeling might look in practice, without all the tangents and sidebars of
the notebook. 
We suggest you start there first, then revisit this file to see how it all comes together.

This file does two things: 
1. It defines a `main()` function, which fits multiple models and produces projections under multiple scenarios.
2. It runs the `main()` function.
The rationale of doing this is so the function could potentially be imported by another script or process and run from there.

Things to remember when running this script:
- Run this script from the covid_model base directory
- make sure the `gcp_project` and `GOOGLE_APPLICATION_CREDENTIALS` environment variables are set appropriately

"""


def main():
    """Here's the main function, all the code we want run will live in this function

    Returns: None

    """
    ####################################################################################################################
    # Set Up

    outdir = setup(os.path.basename(__file__))

    fit_args = {
        'fit_start_date': None,
        'fit_end_date': None,
        'tc_min': 0.0,
        'tc_max': 0.999,
        'tc_window_size': 14,
        'tc_window_batch_size': 5,
        'tc_batch_increment': 2,
        'last_tc_window_min_size': 14,
        'outdir': outdir
    }
    # This set of base model arguments will be used for each scenario we are fitting below
    base_model_args = {
        'params_defs': json.load(open('covid_model/analysis/00000000_hello_world/base_params_for_scenarios.json')),
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',
        'regions': ['co'],
        'mobility_mode': None,
        'start_date': '2020-01-24',
        'end_date': '2022-09-15',
        'max_step_size': 1.0,
        'ode_method': 'RK45'
    }
    # How many different Python processes should we run at a time?
    multiprocess = 6

    # log the above arguments to the log file for the record
    logging.debug(json.dumps({"fit_args": fit_args}, default=str))
    logging.debug(json.dumps({"base_model_args": base_model_args}, default=str))

    ####################################################################################################################
    # Build scenarios and fit models
    logging.info('Compiling scenarios list')

    # Build the list of additional parameters that will be added to the base model arguments to create the individual scenarios
    # notice how we specify the tags to reflect the scenario being run
    scenario_args_list = []
    for beta_mult in [1.00, 0.95, 1.05]:
        for (weak_escape, strong_escape) in [(0.75, 0.1), (0.8, 0.2)]:
            weak_param = [{"param": "immune_escape", "from_attrs": {"immun": "weak", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": weak_escape},  "desc": "weak"}]
            strong_param = [{"param": "immune_escape", "from_attrs": {"immun": "strong", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": strong_escape},  "desc": "strong"}]
            beta_param_adjustment = [{"param": "betta",  "attrs": {"variant": "ba45"}, "mults":  {"2020-01-01": beta_mult}, "desc": "sensitivity"}]
            # these args will literally be passed as arguments when constructing the model for each scenario.
            scenario_args_list.append({'params_defs': base_model_args['params_defs'] + beta_param_adjustment + weak_param + strong_param,
                                       'tags': {'beta_mult': beta_mult, 'ba45_escape_weak': weak_escape, 'ba45_escape_strong': strong_escape},
                                       'hosp_reporting_frac': {"2020-01-01": 1, "2022-03-10": 0.8}})

    # run the scenarios, using the do_fit_scenarios function. This will fit several scenarios at a time, as specified by the `multiprocess` argument
    # even though the model end dates are 9/15, fitting will only occur for dates where hospitalization data is available.
    logging.info('Running Scenarios')
    models = do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess=multiprocess)

    # now that we've fit all the models, let's run a projection for each one.
    # This amounts to running the ODEs forward all the way until the model end dates, using the fitted tc values
    # unless you specify otherwise with the model.update_tc function, the last fitted tc value will persist for all the projected dates

    logging.info('Projecting')
    for model in models:
        logging.info(f'{str(model.tags)}: Projecting')
        model.solve_seir()
        # save the model output, grouping by seir stats, variant, and immunity status
        model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_seir_variant_immun_total_all_at_once_projection.csv')
        # save the model output, keeping all compartments separate
        model.solution_sum_df().unstack().to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'states_full_projection.csv')
        # produce some plots showing the projection
        logging.info(f'{str(model.tags)}: Creating plots')
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

    # create a report for each model, meaning just produce a collection of plots and output files for each model.
    logging.info('Running reports')
    do_create_multiple_reports(models, multiprocess=multiprocess, outdir=outdir, prep_model=False, solve_model=True, immun_variants=['ba2121', 'ba45'], from_date='2022-01-01')


if __name__ == "__main__":
    main()
