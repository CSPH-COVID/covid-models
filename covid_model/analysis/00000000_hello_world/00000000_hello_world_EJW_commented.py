# every file for our covid-models project will begin with this block of code where we set up and load packages
""" Python Standard Library """
# these are packages that are built into Python
import os
import datetime as dt
import json
import logging

""" Third Party Imports """
# these are packages that are built by other people that enhance the software's functionality
# this is one of the many things that make Python great, because it's open source!
from matplotlib import pyplot as plt

""" Local Imports """
# this is code sourced from other files in our code base that we wrote
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
   Pretty much this whole program is embedded inside the definition of the main() function, except for the last two lines.
2. It runs the `main()` function (the last two lines of the program).

It seems a little weird to do this this way, but when you think about it, it can come in handy because then we can
import the function into another script and run it from there. This is called "functional programming."

Things to remember when running this script:
- Run this script from the covid_model base directory
- make sure the `gcp_project` and `GOOGLE_APPLICATION_CREDENTIALS` environment variables are set appropriately

"""

def main():

    """
    Here's the main function, all the code we want run will live in this function

    Args: Document what the arguments are
    Returns: Document what the returns are

    """
    ####################################################################################################################
    # SETUP

    # to set up the logging file, we use the setup() function
    # this function returns a string, which points to the "outdir" variable
    # we will pass this "outdir" variable through both the fitting and report functions
    # doing so will tell these two functions where to save output

    outdir = setup(os.path.basename(__file__))

    # now let's define some Python dictionaries
    # Python dictionaries are always defined by curly brackets {}
    # each entry to the dictionary is a pair (called a "key-value pair")
    # the "key" is a string that identifies what we're looking for in the dictionary (like a dictionary word)
    # the "value" is the data that you want to store (like a dictionary definition)

    # for example, here is the fit_args dictionary, which contains the values that define how we do the fit
    # the keys are the different options we have when doing our fit
    # the values are what those options can be (a value can be any Python object, like a list)

    fit_args = {
        'fit_start_date': None, # "None" (with a capital N) is a special word in Python
                                # in this case, "None" is used to indicate that we are not setting a requirement
                                # for the start date of model fitting (it will pretty much always be this way)
                                # therefore, the fitting function will automatically start on the model start date
        'fit_end_date': None,   # same for end date, but if the model is broken and we need to troubleshoot it, we can
                                # set the value for fit_end_date as May 2nd, 2020, allowing us to fit just two batches
                                # and thus avoiding having to fit the whole thing
        'tc_min': 0.0, # minimum TC that we allow the model to try when fitting (it will pretty much always be this
                       # way, with the exception having been during the Omicron wave when we had to allow TC to be
                       # negative because cases were just so high
        'tc_max': 0.999, # maximum TC that we allow the model to try when fitting
        'tc_window_size': 14, # TC only changes every 14 days
        'tc_window_batch_size': 5, # we fit five TC values at a time
        'tc_batch_increment': 2, # after fitting five batches, we move two batches over
        'last_tc_window_min_size': 14, # fitting has to stop with at least 14 days left to go in the hospital data
                                       # there can be as many as 27 days left, at 28 it will just fit another TC value
        'outdir': outdir # notice that instead of a literal, the value to the 'outdir' key is a variable
                         # pass this variable through the fitting function so that it knows where to save output
    }

    # this set of base model arguments will be used for each scenario we are fitting below
    # instead of defining how we're supposed to do the fit, they define how the model behaves
    # these arguments are shared by all the scenarios that will be run
    base_model_args = {

    # json files are files with a bunch of data in a certain structure that tells us the way it is (like a csv)
    # however, json files are more flexible in their structure (you can nest things, list things out, etc.)
    # these three key-value pairs tell the model where to get data (that doesn't already come from database)
    # the other data that comes from the database is written as SQL queries (you can find it in the project tree)

        # params_defs is just a giant file that has all the parameters (i.e. a number used in an ODE somewhere)
        # could be anything from population to how many people are receiving monoclonal antibodies
        # can be constant or vary over time depending on the parameter
        'params_defs': json.load(open('covid_model/analysis/00000000_hello_world/permanent_params.json')),

        # region_defs defines the regions that we're using for the model
        'region_defs': 'covid_model/input/region_definitions.json',

        # this file defines the vaccine projection parameters
        # if we're projecting for the future, we need to make some sort of assumption
        # as to how vaccinations will look in the future
        # we mostly haven't touched it but we still need to tell the model how to do it
        'vacc_proj_params': 'covid_model/input/20220718_vacc_proj_params.json',

        # define the regions we're working with in this model
        'regions': ['co'], # if we wanted to make this a list of regions, you could say 'regions': ['co', 'ar', 'bo']
                           # lists in Python have square brackets

        # mobility comes into play when there are multiple regions
        # currently set to None because there's only one region right now
        # if we were incorporating mobility, the two other options are 'location_attached' and 'population_attached'
        'mobility_mode': None,

        # model start and end date (self explanatory)
        # relevant because the model has to build its big set of parameters and define the differential equations
        # we want to be smart and build only the ODE between the start and end date
        'start_date': '2020-01-24',
        'end_date': '2022-09-15',

        # technical terms for how the ODE gets solved
        'max_step_size': 1.0,
        'ode_method': 'RK45'
    }

    # multiprocessing spreads out the Python runs over the number of processors you have
    # for example, let's say you have six processes to run, and your computer has three processors
    # if you set multiprocess = 3, it will run three processes at a time and do that two times
    # if you set multiprocess = None, it will run one process at a time
    multiprocess = None

    # log the above arguments to the log file for the record
    # logging messages are structured like this:
    # logging.debug messages won't show up in the red text window, but they will show up in the logging file
    # it keeps everything cleaner but ensures you can go back to the logging file if you need to
    logging.debug(json.dumps({"fit_args": fit_args}, default=str))
    logging.debug(json.dumps({"base_model_args": base_model_args}, default=str))

    ####################################################################################################################
    # BUILD SCENARIOS AND FIT MODELs

    # logging.info and logging.error will make the messages show up both in the file and in the console
    logging.info('Compiling scenarios list')

    # when we run special scenarios, we make those changes in-line
    # this is so that we don't have to mess with that huge base_params file and forget what we changed
    # notice how we specify the tags to reflect the scenario being run, for example...
    # the first tuple (0.75, 0.1) is for the low immune escape scenario
    # the second tuple (0.8, 0.2) is for the high immune escape scenario
    # the square brackets [(0.75, 0.1), (0.8, 0.2)] is called a "list" and it has both these scenarios in them
    # within each tuple, the first number is the immune escape for someone with weak immunity
    # so in this case, 0.75 for low immune escape and 0.8 for high immune escape
    # the other scenario is for strong immunity (immune escape goes way down to 0.1 and 0.2)
    # in THIS code with BA.4/5, between people with weak and strong immunity, not only are people with weak immunity
    # less immune to begin with, but new variants can infect them more easily

    # initialize an empty list for the for-loop
    scenario_args_list = []

    # outer loop changes the multiplier on beta, applied specifically for BA.4/5
    for beta_mult in [1.00, 0.95, 1.05]: # sensitivity analysis for beta to determine how sensitive the model is

        # inner loop goes through the two different scenarios for immune escape for BA.4/5 (low, high)
        for (weak_escape, strong_escape) in [(0.75, 0.1), (0.8, 0.2)]:

            # the loop is just building up a list of scenario-specific model arguments
            # when each scenario is run, it takes the base arguments and then adds on the scenario-specific arguments
            # anything in the scenario-specific parameters that is already in the base parameters
            # but has a different value will overwrite the base parameters
            weak_param = [{"param": "immune_escape", "from_attrs": {"immun": "weak", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": weak_escape},  "desc": "weak"}]
            strong_param = [{"param": "immune_escape", "from_attrs": {"immun": "strong", "variant": ["none", "wildtype", "alpha", "delta", "omicron", "ba2"]}, "to_attrs": {"variant": ["ba45"]},  "vals": {"2020-01-01": strong_escape},  "desc": "strong"}]
            beta_param_adjustment = [{"param": "betta",  "attrs": {"variant": "ba45"}, "mults":  {"2020-01-01": beta_mult}, "desc": "sensitivity"}]

            # use the append() function to add another dictionary to the base model params list
            # what we're appending resembles the base model arguments above
            # take the 'params_defs' from the base_model_args and add three new things
            scenario_args_list.append({'params_defs': base_model_args['params_defs'] + beta_param_adjustment + weak_param + strong_param,
                                       'tags': {'beta_mult': beta_mult, 'ba45_escape_weak': weak_escape, 'ba45_escape_strong': strong_escape},
                                       'hosp_reporting_frac': {"2020-01-01": 1, "2022-03-10": 0.8}})

    # up to this point, we haven't fit the model yet
    # we've just defined a list where each element of the list is a dictionary defining the model arguments

    # run the scenarios, using the do_fit_scenarios function
    # this will fit several scenarios at a time, as specified by the `multiprocess` argument
    # even though the model end date is 9/15, fitting will only occur for dates where hospitalization data is available
    logging.info('Running Scenarios')

    # this is the part where the actual fitting takes place, with the do_fit_scenarios function
    # this function fits all the models and then returns a list of models that have been fit
    # the "models" variable is a list of all the models that have been fit (one for each scenario)
    models = do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess=multiprocess)

    # now that we've fit all the models, let's run a projection for each one
    # this amounts to running the ODE's forward all the way until the model end dates, using the fitted tc values
    # unless you specify otherwise with the model.update_tc function,
    # the last fitted tc value will persist for all the projected dates

    # two more things with this model: we plot some stuff, and we run the report
    # the report spits out all the outputs we use for all our stuff like the out2.csv, immunity plots, forecast, etc.

    # write the logging messages
    logging.info('Projecting')

    # make projections out to the end date
    # plotting is done in matplotlib
    for model in models:
        logging.info(f'{str(model.tags)}: Projecting')
        # use the solve_seir() function to solve the ODE's
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

    # create a report for each model, meaning just produce a collection of plots and output files for each model
    logging.info('Running reports')

    # this function was already defined as well (convenience function)
    # it runs the report for all the models in the "models" list
    do_create_multiple_reports(models, multiprocess=multiprocess, outdir=outdir, prep_model=False, solve_model=True, immun_variants=['ba2121', 'ba45'], from_date='2022-01-01')

# and that's our function! You'll notice it was huuuuuuuuuge (over 200 lines)
# you can tell because it was indented all the way down)

# now we use the last two lines of this program to run the function
if __name__ == "__main__":
    main()
