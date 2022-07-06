""" Python Standard Library """
import copy
import os
from multiprocessing import Pool
import logging
import json
from time import perf_counter
import datetime as dt
""" Third Party Imports """
from multiprocessing_logging import install_mp_handler
import pandas as pd
from scipy import optimize as spo
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
""" Local Imports """
from covid_model import CovidModel
from covid_model.analysis.charts import plot_transmission_control
from covid_model.utils import IndentLogger, setup, get_filepath_prefix, db_engine
from covid_model.analysis.charts import plot_modeled, plot_observed_hosps, format_date_axis
logger = IndentLogger(logging.getLogger(''), {})


def __single_batch_fit(model: CovidModel, tc_min, tc_max, yd_start=None, tstart=None, tend=None, regions=None):
    """function to fit TC for a single batch of time for a model

    Only TC values which lie in the specified regions between tstart and tend will be fit.

    Args:
        model: model to fit
        tc_min: minimum allowable TC
        tc_max: maximum allowable TC
        yd_start: initial conditions for the model at tstart. If None, then model's y0_dict is used.
        tstart: start time for this batch
        tend: end time for this batch
        regions: regions which should be fit. If None, all regions will be fit

    Returns: Fitted TC values and the estimated covariance matrix between the different TC values.

    """
    # define initial states
    regions = model.regions if regions is None else regions
    tc = {t: model.tc[t] for t in model.tc.keys() if tstart <= t <= tend}
    tc_ts  = list(tc.keys())
    yd_start = model.y0_dict if yd_start is None else yd_start
    y0 = model.y0_from_dict(yd_start)
    trange = range(tstart, tend+1)
    ydata = model.hosps.loc[pd.MultiIndex.from_product([regions, [model.t_to_date(t) for t in trange]])]['estimated_actual'].to_numpy().flatten('F')

    def tc_list_to_dict(tc_list):
        """convert tc output of curve_fit to a dict like in our model.

        curve_fit assumes you have a function which accepts a vector of inputs. So it will provide guesses for TC as a
        vector. We need to convert that vector to a dictionary in order to update the model.

        Args:
            tc_list: the list of tc values to update.

        Returns: dictionary of TC suitable to pass to the model.

        """
        i = 0
        tc_dict = {t: {} for t in tc_ts}
        for tc_t in tc.keys():
            for region in regions:
                tc_dict[tc_t][region] = tc_list[i]
                i += 1
        return tc_dict

    def func(trange, *test_tc):
        """A simple wrapper for the model's solve_seir method so that it can be optimzed by curve_fit

        Args:
            trange: the x values of the curve to be fit. necessary to match signature required by curve_fit, but not used b/c we already know the trange for this batch
            *test_tc: list of TC values to try for the model

        Returns: hospitalizations for the regions of interest for the time periods of interest.

        """
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


def do_single_fit(tc_0=0.75,
                  tc_min=0,
                  tc_max=0.99,
                  tc_window_size=14,
                  tc_window_batch_size=6,
                  tc_batch_increment=2,
                  last_tc_window_min_size=21,
                  fit_start_date=None,
                  fit_end_date=None,
                  prep_model=True,
                  outdir=None,
                  write_results=True,
                  write_batch_results=False,
                  model_class=CovidModel,
                  **model_args):
    """ Fits TC for the model between two dates, and does the fit in batches to make the solving easier

    Args:
        tc_0: default value for TC
        tc_min: minimum allowable TC
        tc_max: maximum allowable TC
        tc_window_size: How often to update TC (days)
        tc_window_batch_size: How many windows to fit at once
        tc_batch_increment: How many TC windows to shift over for each batch fit
        last_tc_window_min_size: smallest size of the last TC window
        fit_start_date: refit all tc's on or after this date (if None, use model start date)
        fit_end_date: refit all tc's up to this date (if None, uses either model end date or last date with hospitalization data, whichever is earlier)
        prep_model: Should we run model.prep before fitting (useful if the model_args specify a base_model which has already been prepped)
        outdir: the output directory for saving results
        write_results: should final results be written to the database
        write_batch_results: Should we write output to the database after each fit
        model_class: What class to use for the CovidModel. Useful if using different versions of the model (with more or fewer compartments, say)
        **model_args: Arguments to be used in creating the model to be fit.

    Returns: Fitted model

    """
    def forward_sim_plot(model):
        """Solve the model's ODEs and plot transmission control, and save hosp & TC data to disk

        Args:
            model: the model to solve and plot.
        """
        # TODO: refactor into charts.py?
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

    model = model_class(**model_args)

    # adjust fit start and end, and check for consistency with model and hosp dates
    fit_start_date = dt.datetime.strptime(fit_start_date, '%Y-%m-%d').date() if fit_start_date is not None and isinstance(fit_start_date, str) else fit_start_date
    fit_end_date = dt.datetime.strptime(fit_end_date, '%Y-%m-%d').date() if fit_end_date is not None and isinstance(fit_end_date, str) else fit_end_date
    fit_start_date = model.start_date if fit_start_date is None else fit_start_date
    fit_end_date = min(model.end_date, model.hosps.index.get_level_values(1).max()) if fit_end_date is None else fit_end_date
    ermsg = None
    if fit_start_date < model.start_date:
        ermsg = f'Fit needs to start on or after model start date. Opted to start fitting at {fit_start_date} but model start date is {model.start_date}'
    elif fit_end_date > model.end_date:
        ermsg = f'Fit needs to end on or before model end date. Opted to stop fitting at {fit_end_date} but model end date is {model.end_date}'
    elif fit_end_date > model.hosps.index.get_level_values(1).max():
        ermsg = f'Fit needs to end on or before last date with hospitalization data. Opted to stop fitting at {fit_end_date} but last date with hospitalization data is {model.hosps.index.get_level_values(1).max()}'
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
    if tc_0 is not None:
        tc = {t: tc for t, tc in model.tc.items() if t < fit_tstart or t > fit_tend}
        tc.update({t: {region: tc_0 for region in model.regions} for t in range(fit_tstart, fit_tend - last_tc_window_min_size, tc_window_size)})
        model.update_tc(tc)

    # Get start/end for each batch
    relevant_tc_ts = [t for t in model.tc.keys() if fit_tstart <= t <= fit_tend]
    last_batch_start_index = -min(tc_window_batch_size, len(relevant_tc_ts))
    batch_tstarts =  relevant_tc_ts[:last_batch_start_index:tc_batch_increment] + [relevant_tc_ts[last_batch_start_index]]
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
    """Wrapper function for the do_single_fit function that is useful for parallel / multiprocess fitting.

    Two things are necessary here. First, a new logger needs to be created since this wrapper will run in a new process,
    and second, the model must not write results to the database yet. In order to ensure two models aren't given the same
    spec_id, we have to be careful to write to the database serially. So all the models which were fit in parallel will
    be written to the db one at a time after they are all fit.

    Args:
        args: dictionary of arguments for do_single_fit

    Returns: fitted model that is returned by do_single_fit

    """
    setup(os.path.basename(__file__), 'info')
    logger = IndentLogger(logging.getLogger(''), {})
    return do_single_fit(**args, write_results=False)


def do_single_fit_wrapper_nonparallel(args):
    """Wrapper function for the do_single_fit function that is useful for doing multiple fits serially

    This function can be easily mapped to a list of arguments in order to fit several models in succession

    Args:
        args: dictionary of arguments for do_single_fit

    Returns: fitted model that is returned by do_single_fit

    """
    return do_single_fit(**args, write_results=False)


def do_multiple_fits(model_args_list, fit_args, multiprocess = None):
    """Performs multiple model fits, based on a list of model arguments, all using the same fit_args.

    This function can perform fits in parallel or serially, based on the value of multiprocess. This should work cross-
    platform

    Args:
        model_args_list: list of model_args dictionaries, each of which can be used to construct a model
        fit_args: dictionary of fit arguments that will be applied to each model.
        multiprocess: positive integer indicating how many parallel processes to use, or None if fitting should be done serially

    Returns: list of fitted models, order matching the order of models

    """
    # generate list of arguments
    fit_args2 = {key: val for key, val in fit_args.items() if key not in ['write_results', 'write_batch_output']}
    args_list = list(map(lambda x: {**x, **fit_args}, model_args_list))
    # run each scenario
    if multiprocess:
        #install_mp_handler()  # current bug in multiprocessing-logging prevents this from working right now
        p = Pool(multiprocess)
        models = p.map(do_single_fit_wrapper_parallel, args_list)
    else:
        models = list(map(do_single_fit_wrapper_nonparallel, args_list))

    # write to database serially if specified in those model args
    engine = db_engine()
    if 'write_results' in fit_args and not fit_args['write_results']:
        # default behavior is to write results, so don't write only if specifically told not to.
        pass
    else:
        [m.write_specs_to_db(engine=engine) for m in models]
        [m.write_results_to_db(engine=engine) for m in models]
        logger.info(f'spec_ids: {",".join([str(m.spec_id) for m in models])}')
        logger.info(f'result_ids: {",".join([str(m.result_id) for m in models])}')

    return models


def do_regions_fit(model_args, fit_args, multiprocess=None):
    """Fits a single, disconnected model for each region specified in model_args

    Args:
        model_args: typical model_args used to build a model. Must include list of regions
        fit_args: typical fit_args passed to do_single_fit
        multiprocess: positive int indicating number of parallel processes, or None if fitting should be done serially
    """
    regions = model_args['regions']
    non_region_model_args = {key: val for key, val in model_args.items() if key != 'regions'}
    model_args_list = list(map(lambda x: {'regions': [x], **non_region_model_args, 'tags':{'region': x}}, regions))
    do_multiple_fits(model_args_list, fit_args, multiprocess=multiprocess)


def do_create_report(model, outdir, immun_variants=('ba2121',), from_date=None, to_date=None, prep_model=False, solve_model=False):
    """Create some typically required figures and data for Gov briefings.

    Args:
        model: Model for which to create output
        outdir: path of directory where the output should be saved
        immun_variants: which variants to plot immunity against. Relevant because immune escape factors into immunity and is variant specific
        from_date: start date used in plotting
        to_date: end date used in plotting
        prep_model: whether to run model.prep() before solving the ODEs. useful if model has already been prepped
        solve_model: whether to run model.solve_seir(). useful if model has already been solved.

    Returns: None

    """
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
    """wrapper function for the do_create_report function that can easily be mapped. suitable for creating reports in parallel

    A new logger needs to be created since this wrapper will run in a new process.

    Args:
        args: dictionary of named arguments to be used in do_create_report

    Returns: whatever do_create_report returns

    """
    setup(os.path.basename(__file__), 'info')
    logger = IndentLogger(logging.getLogger(''), {})
    return do_create_report(**args)


def do_create_report_wrapper_nonparallel(args):
    """wrapper function for the do_create_report function that can easily be mapped. suitable for creating reports serially

    Args:
        args: dictionary of named arguments to be used in do_create_report

    Returns: whatever do_create_report returns

    """
    return do_create_report(**args)


def do_create_multiple_reports(models, multiprocess=None, **report_args):
    """Method to easily create multiple reports for various different models. Can be done in parallel or serially

    Args:
        models: list of models that reporst should be created for.
        multiprocess: positive integer indicating how many parallel processes to use, or None if fitting should be done serially
        **report_args: arguments to be passed to do_create_report
    """
    # generate list of arguments
    args_list = list(map(lambda x: {'model': x, **report_args}, models))
    # run each scenario
    if multiprocess:
        #install_mp_handler()  # current bug in multiprocessing-logging prevents this from working right now
        p = Pool(multiprocess)
        p.map(do_create_report_wrapper_parallel, args_list)
    else:
        list(map(do_create_report_wrapper_nonparallel, args_list))


def do_build_legacy_output_df(model: CovidModel):
    """Function to create "legacy output" file, which is a typical need for Gov briefings.

    creates a Pandas DataFrame containing things like prevalence, total infected, and 1-in-X numbers daily for each region

    Args:
        model: Model to create output for.

    Returns: Pandas dataframe containing the output

    """
    totals = model.solution_sum_df(['seir', 'region']).stack(level=1)
    totals['region_pop'] = totals.sum(axis=1)
    totals = totals.rename(columns={'Ih': 'Iht', 'D': 'Dt', 'E': 'Etotal'})
    totals['Itotal'] = totals['I'] + totals['A']

    df = totals.join(model.new_infections).join(model.re_estimates)

    df['prev'] = 100000.0 * df['Itotal'] / df['region_pop']
    df['oneinX'] = df['region_pop'] / df['Itotal']

    return df


def do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess=None):
    """Fits several models using a base set of arguments and a list of scenarios which apply changes to the base settings

    Args:
        base_model_args: dictionary of model args that are common to all scenarios being fit
        scenario_args_list: list of dictionaries, each of which modifies the base_model_args for a particular scenario
        fit_args: fitting arguments applied to all scenarios
        multiprocess: positive integer indicating how many parallel processes to use, or None if fitting should be done serially

    Returns:

    """
    # construct model args from base model args and scenario args list
    model_args_list = []
    for scenario_args in scenario_args_list:
        model_args_list.append(copy.deepcopy(base_model_args))
        model_args_list[-1].update(scenario_args)

    return do_multiple_fits(model_args_list, fit_args, multiprocess)

def do_create_immunity_decay_curves(model, cmpt_attrs):
    print("test")
