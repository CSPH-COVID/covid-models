### Python Standard Library ###
import copy
import os
from multiprocessing import Pool
import logging
import json
from time import perf_counter
import datetime as dt
### Third Party Imports ###
from multiprocessing_logging import install_mp_handler
import numpy as np
import pandas as pd
from scipy import optimize as spo
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
### Local Imports ###
from covid_model import CovidModel, db_engine
from covid_model.analysis.charts import plot_transmission_control
from covid_model.utils import IndentLogger, setup, get_filepath_prefix
from covid_model.analysis.charts import plot_modeled, plot_observed_hosps, format_date_axis
logger = IndentLogger(logging.getLogger(''), {})


def __single_batch_fit(model: CovidModel, tc_min, tc_max, yd_start=None, tstart=None, tend=None):
    # define initial states
    regions = model.regions if regions is None else regions
    tc = {t: model.tc[t] for t in model.tc.keys() if t >= tstart and t <= tend}
    tc_ts  = list(tc.keys())
    yd_start = model.y0_dict if yd_start is None else yd_start
    y0 = model.y0_from_dict(yd_start)
    trange = range(tstart, tend+1)
    ydata = model.hosps.loc[pd.MultiIndex.from_product([regions, [model.t_to_date(t) for t in trange]])]['estimated_actual'].to_numpy().flatten('F')

    # convert tc output by curve_fit to a dict like in our model.
    def tc_list_to_dict(tc_list):
        i = 0
        tc_dict = {t: {} for t in tc_ts}
        for tc_t in tc.keys():
            for region in regions:
                tc_dict[tc_t][region] = tc_list[i]
                i += 1
        return tc_dict

    # function to be optimized
    def func(trange, *test_tc):
        model.update_tc(tc_list_to_dict(test_tc), replace=False, update_lookup=False)
        model.solve_seir(y0=y0, tstart=tstart, tend=tend)
        return model.solution_sum_Ih(tstart, tend, regions=regions)
    fitted_tc, fitted_tc_cov = spo.curve_fit(
        f=func,
        xdata=trange,
        ydata=ydata,
        p0=[tc[t][region] for t in tc_ts for region in model.regions],
        bounds=([tc_min] * len(tc_ts) * len(regions), [tc_max] * len(tc_ts) * len(regions)))
    fitted_tc = tc_list_to_dict(fitted_tc)
    return fitted_tc, fitted_tc_cov

# fits TC for the model between two dates, and does the fit in batches to make the solving easier
def do_single_fit(tc_0=0.75,  # default value for TC
                  tc_min=0,  # minimum allowable TC
                  tc_max=0.99,  # maximum allowable TC
                  tc_window_size=14,  # How often to update TC (days)
                  tc_window_batch_size=5,  # How many windows to fit at once
                  tc_batch_increment=1,  # How many TC windows to shift over for each batch fit
                  last_tc_window_min_size=21,  # smallest size of the last TC window
                  fit_start_date=None,  # refit all tc's on or after this date (if None, use model start date)
                  fit_end_date=None,  # refit all tc's up to this date (if None, uses either model end date or last date with hospitalization data, whichever is earlier
                  prep_model=True,  # Should we run model.prep
                  outdir=None,  # the output directory for saving results
                  write_results=True,  # should final results be written to the database
                  write_batch_results=False,  # Should we write output to the database after each fit
                  **model_args):

    def forward_sim_plot(model):
        # TODO: refactor into charts?
        logger.info(f'{str(model.tags)}: Running forward sim')
        fig, axs = plt.subplots(2, len(model.regions), figsize=(10*len(model.regions), 10), dpi=300, sharex=True, sharey=False, squeeze=False)
        hosps_df = model.modeled_vs_observed_hosps()
        for i, region in enumerate(model.regions):
            hosps_df.loc[region].plot(ax=axs[0, i])
            axs[0,i].title.set_text(f'Hospitalizations: {region}')
            plot_transmission_control(model, [region], ax=axs[1,i])
            axs[1, i].title.set_text(f'TC: {region}')
        plt.savefig(get_filepath_prefix(outdir, tags=model.tags) + '_model_fit.png')
        plt.close()
        hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_fit.csv')
        json.dump(dict(model.tc), open(get_filepath_prefix(outdir, tags=model.tags) + '_model_tc.json', 'w'))

    logging.debug(str({"model_build_args": model_args}))

    # get a db connection if we're going to be writing results
    if write_batch_results or write_results:
        engine = db_engine()

    model = CovidModel(**model_args)

    # adjust fit start and end, and check for consistency with model and hosp dates
    fit_start_date = model.start_date if fit_start_date is None else fit_start_date
    fit_end_date = min(model.end_date, model.hosps.index.get_level_values(1).max()) if fit_end_date is None else fit_end_date
    ermsg = None
    if fit_start_date < model.start_date:
        ermsg = f'Fit needs to start on or after model start date. Opted to start fitting at {fit_start_date} but model start date is {model.start_date}'
    elif fit_end_date > model.end_date:
        ermsg = f'Fit needs to end on or before model end date. Opted to stop fitting at {fit_end_date} but model end date is {model.end_date}'
    elif fit_end_date > model.hosps.index.get_level_values(1).max():
        ermsg = f'Fit needs to end on or before last date with hospitalization data. Opted to stop fitting at {fit_end_date} but last date with hospitalization data is {model.observed_hosp.index.max()}'
    if ermsg is not None:
        logger.exception(f"{str(model.tags)}" + ermsg)
        raise ValueError(ermsg)

    # prep model (we only do this once to save time)
    if prep_model:
        logger.info(f'{str(model.tags)} Prepping Model')
        t0 = perf_counter()
        model.prep(outdir=outdir)
        logger.debug(f'{str(model.tags)} Model flows {model.flows_string}')
        logger.info(f'{str(model.tags)} Model prepped for fitting in {perf_counter() - t0} seconds.')

    # replace the TC and tslices within the fit window
    fit_tstart = model.date_to_t(fit_start_date)
    fit_tend = model.date_to_t(fit_end_date)
    tc = {t: tc for t, tc in model.tc.items() if t < fit_tstart or t > fit_tend}
    if tc_0 is not None:
        tc.update({t: {region: tc_0 for region in model.regions} for t in range(fit_tstart, fit_tend - last_tc_window_min_size, tc_window_size)})
        model.update_tc(tc)

    # Get start/end for each batch
    relevant_tc_ts = [t for t in model.tc.keys() if t >= fit_tstart and t <= fit_tend]
    batch_tstarts =  relevant_tc_ts[:-tc_window_batch_size:tc_batch_increment] + [relevant_tc_ts[-tc_window_batch_size]]
    batch_tends = [t - 1 for t in relevant_tc_ts[tc_window_batch_size::tc_batch_increment]] + [fit_tend]

    logger.info(f'{str(model.tags)} Will fit {len(batch_tstarts)} times')
    for i, (tstart, tend) in enumerate(zip(batch_tstarts, batch_tends)):
        t0 = perf_counter()
        yd_start = model.y_dict(tstart) if tstart != 0 else model.y0_dict
        fitted_tc, fitted_tc_cov = __single_batch_fit(model, tc_min=tc_min, tc_max=tc_max, yd_start=yd_start, tstart=tstart, tend=tend)
        model.tags['fit_batch'] = str(i)
        logger.info(f'{str(model.tags)}: Transmission control fit {i + 1}/{len(batch_tstarts)} completed in {perf_counter() - t0} seconds: {fitted_tc}')

        if write_batch_results:
            logger.info(f'{str(model.tags)}: Uploading batch results')
            model.write_specs_to_db(engine)
            logger.info(f'{str(model.tags)}: spec_id: {model.spec_id}')

        # simulate the model and save a picture of the output
        forward_sim_plot(model)

    model.tc_cov = fitted_tc_cov
    model.tags['run_type'] = 'fit'
    logger.info(f'{str(model.tags)}: fitted TC: {model.tc}')

    if outdir is not None:
        forward_sim_plot(model)

    if write_results:
        logger.info(f'{str(model.tags)}: Uploading final results')
        model.write_specs_to_db(engine)
        #model.write_results_to_db(engine)
        logger.info(f'{str(model.tags)}: spec_id: {model.spec_id}')

    return model


def do_single_fit_wrapper_parallel(args):
    setup(os.path.basename(__file__), 'info')
    logger = IndentLogger(logging.getLogger(''), {})
    return do_single_fit(**args, write_results=False)


def do_single_fit_wrapper_nonparallel(args):
    return do_single_fit(**args, write_results=False)


def do_multiple_fits(model_args_list, fit_args, multiprocess = None):
    # generate list of arguments
    fit_args2 = {key: val for key, val in fit_args.items() if key not in ['write_results', 'write_batch_output']}
    args_list = list(map(lambda x: {**x, **fit_args}, model_args_list))
    # run each scenario
    if multiprocess:
        install_mp_handler()
        p = Pool(multiprocess)
        models = p.map(do_single_fit_wrapper_parallel, args_list)
    else:
        models = list(map(do_single_fit_wrapper_nonparallel, args_list))

    # write to database serially
    engine = db_engine()
    [m.write_specs_to_db(engine=engine) for m in models]
    [m.write_results_to_db(engine=engine) for m in models]
    logger.info(f'spec_ids: {",".join([str(m.spec_id) for m in models])}')
    logger.info(f'result_ids: {",".join([str(m.result_id) for m in models])}')

    return models


def do_regions_fit(model_args, fit_args, multiprocess=None):
    regions = model_args['regions']
    non_region_model_args = {key: val for key, val in model_args.items() if key != 'regions'}
    model_args_list = list(map(lambda x: {'regions': [x], **non_region_model_args, 'tags':{'region': x}}, regions))
    do_multiple_fits(model_args_list, fit_args, multiprocess=multiprocess)


def do_create_report(model, outdir, immun_variants=('ba2121'), from_date=None, to_date=None, prep_model=False, solve_model=False):
    from_date = model.start_date if from_date is None else from_date
    from_date = dt.datetime.strptime(from_date, '%Y-%m-%d').date() if isinstance(from_date, str) else from_date
    to_date = model.end_date if to_date is None else to_date
    to_date = dt.datetime.strptime(to_date, '%Y-%m-%d').date() if isinstance(to_date, str) else to_date

    if prep_model:
        logger.info('Prepping model')
        t0 = perf_counter()
        model.prep(outdir=outdir)
        t1 = perf_counter()
        logger.info(f'Model prepped in {t1 - t0} seconds.')

    if solve_model:
        logger.info('Solving model')
        model.solve_seir()

    subplots_args = {'figsize': (10, 8), 'dpi': 300}

    # prevalence
    fig, ax = plt.subplots(**subplots_args)
    ax.set_ylabel('SARS-CoV-2 Prevalence')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc='best')
    plot_modeled(model, ['I', 'A'], share_of_total=True, ax=ax, label='modeled')
    format_date_axis(ax)
    ax.set_xlim(from_date, to_date)
    ax.axvline(x=dt.date.today(), color='darkgray')
    ax.grid(color='lightgray')
    ax.legend(loc='best')
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'prevalence.png')
    plt.close()

    # hospitalizations
    #TODO: update to be the back_adjusted hosps
    fig, ax = plt.subplots(**subplots_args)
    ax.set_ylabel('Hospitalized with COVID-19')
    plot_observed_hosps(db_engine(), ax=ax, color='black')
    plot_modeled(model, 'Ih', ax=ax, label='modeled')
    format_date_axis(ax)
    ax.set_xlim(from_date, to_date)
    ax.axvline(x=dt.date.today(), color='darkgray')
    ax.grid(color='lightgray')
    ax.legend(loc='best')
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'hospitalized.png')
    plt.close()

    # variants
    fig, ax = plt.subplots(**subplots_args)
    plot_modeled(model, ['I', 'A'], groupby='variant', share_of_total=True, ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel('Variant Share of Infections')
    format_date_axis(ax)
    ax.set_xlim(from_date, to_date)
    ax.axvline(x=dt.date.today(), color='darkgray')
    ax.grid(color='lightgray')
    ax.legend(loc='best')
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'variant_share.png')
    plt.close()

    # immunity
    for variant in immun_variants:
        fig, ax = plt.subplots(**subplots_args)
        immun = model.immunity(variant=variant)
        immun_65p = model.immunity(variant=variant, age='65+')
        immun_hosp = model.immunity(variant=variant, to_hosp=True)
        immun_hosp_65p = model.immunity(variant=variant, age='65+', to_hosp=True)
        for df, name in zip((immun, immun_65p, immun_hosp, immun_hosp_65p), ('immun', 'immun_65p', 'immun_hosp', 'immun_hosp_65p')):
            df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + f'{name}_{variant}.csv')
        ax.plot(model.daterange, immun, label=f'Immunity vs Infection', color='cyan')
        ax.plot(model.daterange, immun_65p, label=f'Immunity vs Infection (65+ only)', color='darkcyan')
        ax.plot(model.daterange, immun_hosp, label=f'Immunity vs Severe Infection', color='gold')
        ax.plot(model.daterange, immun_hosp_65p, label=f'Immunity vs Severe Infection (65+ only)', color='darkorange')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Percent Immune')
        format_date_axis(ax)
        ax.set_xlim(from_date, to_date)
        ax.axvline(x=dt.date.today(), color='darkgray')
        ax.grid(color='lightgray')
        ax.legend(loc='best')
        fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + f'immunity_{variant}.png')
        plt.close()

    do_build_legacy_output_df(model).to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'out2.csv')

    return None


def do_create_report_wrapper_parallel(args):
    setup(os.path.basename(__file__), 'info')
    logger = IndentLogger(logging.getLogger(''), {})
    return do_create_report(**args)


def do_create_report_wrapper_nonparallel(args):
    return do_create_report(**args)


def do_create_multiple_reports(models, multiprocess=None, **report_args):
    # generate list of arguments
    args_list = list(map(lambda x: {'model': x, **report_args}, models))
    # run each scenario
    if multiprocess:
        install_mp_handler()
        p = Pool(multiprocess)
        p.map(do_create_report_wrapper_parallel, args_list)
    else:
        list(map(do_create_report_wrapper_nonparallel, args_list))


def do_build_legacy_output_df(model: CovidModel):
    totals = model.solution_sum_df(['seir', 'region']).stack(level=1)
    totals['region_pop'] = totals.sum(axis=1)
    totals = totals.rename(columns={'Ih': 'Iht', 'D': 'Dt', 'E': 'Etotal'})
    totals['Itotal'] = totals['I'] + totals['A']

    df = totals.join(model.new_infections).join(model.re_estimates)

    df['prev'] = 100000.0 * df['Itotal'] / df['region_pop']
    df['oneinX'] = df['region_pop'] / df['Itotal']

    return df


def do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess = None):
    # construct model args from base model args and scenario args list
    model_args_list = []
    for scenario_args in scenario_args_list:
        model_args_list.append(copy.deepcopy(base_model_args))
        model_args_list[-1].update(scenario_args)

    return do_multiple_fits(model_args_list, fit_args, multiprocess)
