#!/usr/bin/env python
import scipy.integrate

if __name__ == "__main__":
    """ Python Standard Library """
    import os
    import datetime as dt
    import json
    import logging
    import pickle

    """ Third Party Imports """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns

    """ Local Imports """
    if 'requirements.txt' not in os.listdir(os.getcwd()):
        os.chdir(os.path.join('../../../../..', '..', '..'))
    print(os.getcwd())
    # Import the RMW model instead of the original model
    from covid_model.rmw_model import CovidModel as RMWCovidModel
    from covid_model.runnable_functions import do_regions_fit, do_single_fit, do_fit_scenarios, \
        do_create_multiple_reports
    from covid_model.utils import setup, get_filepath_prefix
    from covid_model.analysis.charts import plot_transmission_control

    # os.environ['gcp_project'] = 'co-covid-models'
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "co-covid-models-credentials.json"

    # In[5]:

    # set up the output directory for this Jupyter notebook
    outdir = setup("rmw_scenario_projections.ipynb")

    # ### Fit an initial scenario through February 2022

    # In[6]:

    # designate the arguments for how the model will behave
    model_args = {
        'params_defs': 'covid_model/input/rmw_params_scaled.json',
        'region_defs': 'covid_model/input/rmw_region_definitions.json',
        'vacc_proj_params': 'covid_model/analysis/20221004_oct_gov_briefing/20221004_vacc_proj_params.json',
        'start_date': '2020-01-24',
        'end_date': '2024-01-01',
        'regions': ['coe', 'con', 'cow']
    }

    # this is how the fit will behave
    # place the outdir argument here to tell the model fit where to go
    fit_args = {'outdir': outdir,
                'fit_end_date': '2022-02-28',
                'model_class': RMWCovidModel
                }

    # because all the scenarios are the same
    # List of regions to use
    regions_to_use = ["con"]
    scen_args = [{"regions": [region]} for region in regions_to_use]
    #model = do_single_fit(**model_args,**fit_args)
    models = do_fit_scenarios(base_model_args=model_args,
                              fit_args=fit_args,
                              scenario_args_list=scen_args)

    for reg, reg_model in zip(regions_to_use, models):
        with open(f"solution_df_{reg}_new.pkl", "wb") as f:
            pickle.dump(reg_model.solution_ydf, f)
    exit(0)
    multiprocess = 4

    for reg, reg_model in zip(regions_to_use, models):
        with open(f"solution_df_{reg}_new.pkl", "wb") as f:
            pickle.dump(reg_model.solution_ydf, f)

        scenario_params = json.load(open("covid_model/input/rmw_params_scaled.json"))

        model_args = {
            'base_spec_id': reg_model.spec_id,  # model.spec_id, # use the spec id that was output from the model fit
            'regions':reg_model.regions
        }
        model_fit_args = {
            'outdir': outdir,
            'fit_start_date': '2022-03-01',
            # set the start date for the earliest point at which the scenarios start to differ from one another
            'pre_solve_model': True,
            # force the model to establish initial conditions so the fit can start on the fit start date
            'model_class': RMWCovidModel
        }

        # define vaccine effectiveness for < 5 (this is a multiplier for the baseline vaccine effectiveness for 0-19)
        vacc_eff_lt5 = 0.5

        # Create different scenarios to model
        scenario_model_args = []
        for vx_seed in [0, 5]:
            for vir_mult in [0.833, 2.38]:
                hrf = {"2020-01-01": 1, "2022-03-01": (0.66 + 0.34 * 0.8),
                       "2022-03-15": (0.34 + 0.66 * 0.8), "2022-03-30": 0.8}
                vx_adjust = [{"param": "vx_seed",
                              "vals": {"2020-01-01": 0, "2022-09-30": vx_seed, "2022-10-30": 0},
                              "desc": "Variant X seeding"}]
                vir_adjust = [{"param": "hosp",
                               "attrs": {"variant": "vx"},
                               "mults": {"2020-01-01": vir_mult},
                               "desc": "Variant X hospitalization multiplier"}]
                lt5_vacc_adjust = [{"param": "immunity",
                                    "attrs": {'age': '0-19', 'vacc': 'shot1'},
                                    "mults": {"2020-01-01": 1,
                                              "2022-06-20": 0.99 + 0.01 * vacc_eff_lt5,
                                              "2022-06-30": 0.98 + 0.02 * vacc_eff_lt5,
                                              "2022-07-10": 0.97 + 0.03 * vacc_eff_lt5,
                                              "2022-07-20": 0.96 + 0.04 * vacc_eff_lt5,
                                              "2022-08-10": 0.95 + 0.05 * vacc_eff_lt5,
                                              "2022-08-30": 0.94 + 0.06 * vacc_eff_lt5,
                                              "2022-09-20": 0.93 + 0.07 * vacc_eff_lt5},
                                    "desc": "weighted average using share of 0-19 getting shot1 who are under 5"}]
                scenario_model_args.append({'params_defs': scenario_params + vx_adjust + vir_adjust + lt5_vacc_adjust,
                                            'hosp_reporting_frac': hrf,
                                            'tags': {'vx_seed': vx_seed,
                                                     'vir_mult': vir_mult,
                                                     'booster_mult': 0,
                                                     'region':"_".join(reg_model.regions)},
                                            'regions':reg_model.regions})

        # %%

        # check how many scenarios there are
        len(scenario_model_args)

        # %%

        # run the scenarios
        models = do_fit_scenarios(base_model_args=model_args, scenario_args_list=scenario_model_args, fit_args=model_fit_args)

        # %% md

        ### Run the report for each fit model

        # %%

        # here you can also specify which variants you want to calculate immunity for
        do_create_multiple_reports(models, multiprocess=multiprocess, outdir=outdir, prep_model=False, solve_model=True,
                                   immun_variants=['ba45', 'vx'], from_date='2022-01-01')

        # %%

        logging.info('Projecting')
        for model in models:
            logging.info('')
            # model.prep()  # don't think we need to prep anymore.
            model.solve_seir()

            model.solution_sum_df(['seir', 'variant', 'immun']).unstack().to_csv(
                get_filepath_prefix(outdir, tags=model.tags) + 'states_seir_variant_immun_total_all_at_once_projection.csv')
            model.solution_sum_df().unstack().to_csv(
                get_filepath_prefix(outdir, tags=model.tags) + 'states_full_projection.csv')

            logging.info(f'{str(model.tags)}: Running forward sim')
            fig = plt.figure(figsize=(10, 10), dpi=300)
            ax = fig.add_subplot(211)
            hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
            hosps_df.plot(ax=ax)
            ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(),
                        dt.datetime.strptime('2024-01-01', "%Y-%m-%d").date())
            ax = fig.add_subplot(212)
            plot_transmission_control(model, ax=ax)
            ax.set_xlim(dt.datetime.strptime('2022-01-01', "%Y-%m-%d").date(),
                        dt.datetime.strptime('2024-01-01', "%Y-%m-%d").date())
            plt.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast.png')
            plt.close()
            hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_forecast.csv')
            json.dump(model.tc, open(get_filepath_prefix(outdir, tags=model.tags) + 'model_forecast_tc.json', 'w'))

        logging.info('Running reports')
