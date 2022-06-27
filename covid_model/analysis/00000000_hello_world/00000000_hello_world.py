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
Welcome! This script will demonstrate how to do some basic things with the model. You will need access to the database
as well as all required python packages installed.

This file is heavily commented for your benefit!

This file only does two things: defines a function called main(), then calls the function. This is good practice for a 
Python script, as it makes it possible for another script to load this same main function and run it at will.

The main function shows a few different ways to build and fit models.

"""

# Here's the main function, all the code we want run will live in this function
def main():
    ####################################################################################################################
    # Set Up

    # the `setup` function sets up loggin, as well as determines the output directory for this script.
    # logging enables us to make print statements look prettier and allows for logging to the terminal window and to a text file simultaneously.
    # the output directory `outdir` is based on this file's name, so it's easy to see what output was produced by this script
    # by passing `outdir` to functions below, we ensure all the output from those functions is saved in the same location.
    outdir = setup(os.path.basename(__file__))

    # we can test that the logging is working by trying to log a message like so:
    logging.info("This is a logging message")
    # when you run this file, you should see this message printed to the console and also saved in the log file.
    # By default, debug messages will not be printed to the console but will be saved in the log file:
    logging.debug("You won't see this message in the console, but you will see it in the console")

    # This dictionary defines the arguments to use when fitting the model. There are many options, but they all have
    # defaults, so it's only necessary to specify the non-default options here.
    fit_args = {
        'outdir': outdir                    # Where to save output from fitting
    }

    # This dictionary specifies arguments for the model. The model is an instance of CovidModel, and any attribute of CovidModel can be specified here.
    model_args = {
        'params_defs': 'covid_model/analysis/00000000_hello_world/params.json',     # the file that defines all the parameters for the model
        'region_defs': 'covid_model/input/region_definitions.json',                     # the file that defines all the regions used in the model
        'vacc_proj_params': 'covid_model/input/vacc_proj_params.json',                  # the file that defines how vaccines projections will be made in the event we lack recent vaccination data
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),            # The start date of the model.
        'end_date': dt.datetime.strptime('2021-11-09', "%Y-%m-%d").date(),              # The end date of the model.
        'max_step_size': 1.0,                                                           # The biggest allowable step (in days) that the ODE solver is allowed to take.
        'ode_method': 'RK23'                                                            # what method to use when solving the ODE
    }

    # it's a good idea to log the fit args and the model args we are using
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"model_args": model_args}, default=str))

    ####################################################################################################################
    # Creating a model
    logging.info('Building Models')

    # A model is just an instance of the CovidModel class.
    # Models can be created in several ways, below are some examples:

    # We can create a model using some model arguments
    m1 = CovidModel(**model_args)
    # Note: this is equivalent to writing CovidModel(params_defs = 'covid_model/analysis/00000000_hello_world/params.json', region_defs = ... ) etc.
    # It's nice to define all the model arguments in the dictionary above so we can see them all at a glance and it's easy to dump them to the log file.


    #base_model_args = {'base_spec_id': 2710, 'params_defs': json.load(open('covid_model/analysis/20220606_GovBriefing/params_no_ba45_immuneescape.json'))}


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