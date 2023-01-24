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
from covid_model.analysis.20220620_much_simpler_regional_model.simple_model import SimpleCovidModel
from covid_model.runnable_functions import do_single_fit, do_create_report
from covid_model.utils import setup, get_filepath_prefix
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
    tslices =[0,14,28,42,56,70,84,98,112,126,140,154,168,182,196,210,224,238,252,266,280,294,308,322,336,350,364,378,392,406,420,434,448,462,476,490,504,518,532,546,560,574,588,602,616,630,644,658,672,686,700,714,728,742,756,770,784,798,812,826,840,854]
    tc_vals = [0.914894786879403,0.67314824191622979,0.50426837046140283,8.2520651028438247e-17,0.69829215490592678,0.83677829837381568,0.82868233246820089,0.83221528782592658,0.87702445303921783,0.82394029330960972,0.76202213592379331,0.66963715593978745,0.84426413093504982,0.78071503998822622,0.82138504914934107,0.76999608123053143,0.77307020946796534,0.71860981066622576,0.72277228376241909,0.6825502829383302,0.72195760353608218,0.75746802279235015,0.8231123466598762,0.810848400395522,0.78344599584652208,0.79367853701340318,0.80536372766097741,0.74366007648620946,0.77992226913379326,0.70678224282172364,0.71858551926939207,0.68733874046548449,0.74077715471942207,0.81507154216969191,0.793814817372185,0.83381734963125509,0.79164712504347667,0.78474973392507952,0.7648576289529383,0.71720604418944855,0.74644121899136362,0.75394282714374883,0.79256301042086175,0.77097429343695634,0.7228125736084704,0.741432303848742,0.71582594771093122,0.75599482396089224,0.75839037810054444,0.74053434119565842,0.65197672327021161,0.79980010615402031,0.83300402425058506,0.85077662494564144,0.8432345083710332,0.85499627310229853,0.86282958581519464,0.82653917403805643,0.83261417058134,0.83208524222865277,0.83910355429514849,0.83901912043819038]
    tc = {t: {'co': val} for t, val in zip(tslices, tc_vals)}
    full_model_args = {
        'params_defs': 'covid_model/analysis/20220620_much_simpler_regional_model/more_complicated_model_params.json',
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/20220718_vacc_proj_params.json',
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-10-01', "%Y-%m-%d").date()
    }
    model_args = {
        'params_defs': 'covid_model/analysis/20220620_much_simpler_regional_model/co_local_model_params.json',
        'region_defs': 'covid_model/input/region_definitions.json',
        'vacc_proj_params': 'covid_model/input/20220718_vacc_proj_params.json',
        'mobility_mode': 'population_attached',
        'start_date': dt.datetime.strptime('2020-01-24', "%Y-%m-%d").date(),
        'end_date': dt.datetime.strptime('2022-10-01', "%Y-%m-%d").date(),
        'max_step_size': 1.0
    }
    #base_model_args = {'base_spec_id': 2710, 'params_defs': json.load(open('covid_model/analysis/20220606_GovBriefing/params_no_ba45_immuneescape.json'))}
    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(json.dumps({"model_args": model_args}, default=str))

    ####################################################################################################################
    # Run

    logging.info('Building Models')
    model = CovidModel(**full_model_args, tags={"version": "full"})
    model.update_tc(tc)
    model.prep(outdir=outdir)

    smodel = SimpleCovidModel(**model_args, tags={"version": "simplified_same_tc"})
    smodel.update_tc(tc)
    smodel.prep(outdir=outdir)

    logging.info('Solving Models')
    model.solve_seir()
    smodel.solve_seir()

    logging.info('fitting simplified model')
    # now fit TC for the simpler model and compare to the bigger model.
    smodel2 = do_single_fit(**fit_args, write_results=False, **model_args, tags={'version': 'simplified_fit_tc'}, model_class=SimpleCovidModel)

    logging.info('Creating report')

    do_create_report(model, outdir, immun_variants=['ba45'])
    do_create_report(smodel, outdir, immun_variants=['omicron'])
    do_create_report(smodel2, outdir, immun_variants=['omicron'])


    # Plot solutions to compare differences

    fig, axs = plt.subplots(2, len(model.regions), figsize=(10 * len(model.regions), 10), dpi=300, sharex=True, sharey=False, squeeze=False)
    hosps_df = model.modeled_vs_observed_hosps()
    shosps_df=smodel.modeled_vs_observed_hosps()
    for i, region in enumerate(model.regions):
        hosps_df.loc[region].plot(ax=axs[0, i])
        shosps_df.loc[region].plot(ax=axs[0, i])
        axs[0, i].title.set_text(f'Hospitalizations: {region}')
        plot_transmission_control(model, [region], ax=axs[1, i])
        plot_transmission_control(smodel, [region], ax=axs[1, i])
        axs[1, i].title.set_text(f'TC: {region}')

    # collapse model compartments to the simple model so we can compare their solutions
    attr_replace = {
        'seir': {},
        'age': {},
        'vacc': {'shot1': 'none', 'shot2': 'vaccinated', 'shot3': 'vaccinated'},
        'variant': {'wildtype': 'preomicron', 'alpha': 'preomicron', 'delta': 'preomicron', 'ba2': 'omicron',
                    'ba2121': 'omicron', 'ba45': 'omicron'},
        'immun': {'weak': 'immune', 'strong': 'immune'},
        'region': {}
    }
    simpler_ydf = model.solution_ydf.transpose().reset_index() \
        .replace(attr_replace).groupby(list(attr_replace.keys())).aggregate(sum) \
        .sort_index().transpose().reindex(smodel.solution_ydf.columns, axis=1)






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