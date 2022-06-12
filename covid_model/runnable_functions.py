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
    tc = {t: model.tc[t] for t in model.tc.keys() if t >= tstart and t <= tend}
    tc_ts  = list(tc.keys())
    yd_start = model.y0_dict if yd_start is None else yd_start
    y0 = model.y0_from_dict(yd_start)
    trange = range(tstart, tend+1)

    # function to be optimized
    def func(trange, *test_tc):
        for i, tc_val in enumerate(test_tc):
            tc[tc_ts[i]] = tc_val
        model.update_tc(tc, replace=False, update_lookup=False)
        model.solve_seir(y0=y0, tstart=tstart, tend=tend, method='RK45')
        return model.solution_sum_Ih(tstart, tend)
    fitted_tc, fitted_tc_cov = spo.curve_fit(
        f=func,
        xdata=trange,
        ydata=model.estimated_actual_hosp[trange],
        p0=[tc[t] for t in tc_ts],
        bounds=([tc_min] * len(tc_ts), [tc_max] * len(tc_ts)))
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
                  loss_projection_days=None,  # can compute loss of fit using only the batch window (if =0), or also include a projection (in days) into the future. If None, will project until fit_end_date
                  prep_model=True,  # Should we run model.prep
                  outdir=None,  # the output directory for saving results
                  write_results=True,  # should final results be written to the database
                  write_batch_results=False,  # Should we write output to the database after each fit
                  **model_args):

    def forward_sim_plot(model):
        # TODO: refactor into charts?
        logger.info(f'{str(model.tags)}: Running forward sim')
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(211)
        hosps_df = model.modeled_vs_observed_hosps().reset_index('region').drop(columns='region')
        hosps_df.plot(ax=ax)
        ax = fig.add_subplot(212)
        plot_transmission_control(model, ax=ax)
        plt.savefig(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_fit.png')
        plt.close()
        hosps_df.to_csv(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_fit.csv')
        json.dump(dict(model.tc), open(get_filepath_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_tc.json', 'w'))

    logging.debug(str({"model_build_args": model_args}))

    # get a db connection if we're going to be writing results
    if write_batch_results or write_results:
        engine = db_engine()

    model = CovidModel(**model_args)

    # adjust fit start and end, and check for consistency with model and hosp dates
    fit_start_date = model.start_date if fit_start_date is None else fit_start_date
    fit_end_date = min(model.end_date, model.observed_hosp.index.max()) if fit_end_date is None else fit_end_date
    ermsg = None
    if fit_start_date < model.start_date:
        ermsg = f'Fit needs to start on or after model start date. Opted to start fitting at {fit_start_date} but model start date is {model.start_date}'
    elif fit_end_date > model.end_date:
        ermsg = f'Fit needs to end on or before model end date. Opted to stop fitting at {fit_end_date} but model end date is {model.end_date}'
    elif fit_end_date > model.observed_hosp.index.max():
        ermsg = f'Fit needs to end on or before last date with hospitalization data. Opted to stop fitting at {fit_end_date} but last date with hospitalization data is {model.observed_hosp.index.max()}'
    if ermsg is not None:
        logger.exception(f"{str(model.tags)}" + ermsg)
        raise ValueError(ermsg)

    # prep model (we only do this once to save time)
    if prep_model:
        logger.info(f'{str(model.tags)} Prepping Model')
        t0 = perf_counter()
        model.prep()
        logger.debug(f'{str(model.tags)} Model flows {model.flows_string}')
        logger.info(f'{str(model.tags)} Model prepped for fitting in {perf_counter() - t0} seconds.')

    # replace the TC and tslices within the fit window
    fit_tstart = model.date_to_t(fit_start_date)
    fit_tend = model.date_to_t(fit_end_date)
    tc = {t: tc for t, tc in model.tc.items() if t < fit_tstart or t > fit_tend}
    tc.update({t: tc_0 for t in range(fit_tstart, fit_tend - last_tc_window_min_size, tc_window_size)})
    model.update_tc(tc)

    # Get start/end for each batch
    relevant_tc_ts = [t for t in model.tc.keys() if t >= fit_tstart and t <= fit_tend]
    batch_tstarts =  relevant_tc_ts[:-tc_window_batch_size:tc_batch_increment] + [relevant_tc_ts[-tc_window_batch_size]]
    batch_tends = [t - 1 for t in relevant_tc_ts[tc_window_batch_size::tc_batch_increment]] + [fit_tend]

    logger.info(f'{str(model.tags)} Will fit {len(batch_tstarts)} times')
    for i, (tstart, tend) in enumerate(zip(batch_tstarts, batch_tends)):
        tend = min(fit_tend, tend + loss_projection_days) if loss_projection_days is not None else fit_tend
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
        model.write_results_to_db(engine)
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


def do_create_report(model, outdir, from_date=None, to_date=None, prep_model=False, solve_model=False):
    from_date = model.start_date if from_date is None else from_date
    to_date = model.end_date if to_date is None else to_date

    if prep_model:
        logger.info('Prepping model')
        t0 = perf_counter()
        model.prep()
        t1 = perf_counter()
        logger.info(f'Model prepped in {t1 - t0} seconds.')

    if solve_model:
        logger.info('Solving model')
        model.solve_seir()
    # build_legacy_output_df(model).to_csv('output/out2.csv')

    size_per_chart = 8
    fig, axs = plt.subplots(2,2, figsize=(size_per_chart * 2 + 1, size_per_chart * 2))

    # prevalence
    ax = axs.flatten()[0]
    ax.set_ylabel('SARS-CoV-2 Prevalenca')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc='best')
    plot_modeled(model, ['I', 'A'], share_of_total=True, ax=ax, label='modeled')

    # hospitalizations
    ax = axs.flatten()[1]
    ax.set_ylabel('Hospitalized with COVID-19')
    plot_observed_hosps(db_engine(), ax=ax, color='black')
    plot_modeled(model, 'Ih', ax=ax, label='modeled')

    #hosps_df = pd.DataFrame(index=model.trange)
    #hosps_df['modeled'] = model.solution_sum('seir')['Ih']
    #hosps_df.index = model.daterange
    #hosps_df.loc[:'2022-02-28'].round(1).to_csv(get_filepath_prefix(outdir) + 'omicron_report_hospitalizations.csv')

    # variants
    ax = axs.flatten()[2]
    plot_modeled(model, ['I', 'A'], groupby='variant', share_of_total=True, ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel('Variant Share of Infections')

    # immunity
    ax = axs.flatten()[3]
    ax.plot(model.daterange, model.immunity('omicron'), label='Immunity vs Omicron', color='cyan')
    ax.plot(model.daterange, model.immunity('omicron', age='65+'), label='Immunity vs Omicron (65+ only)', color='darkcyan')
    ax.plot(model.daterange, model.immunity('omicron', to_hosp=True), label='Immunity vs Severe Omicron', color='gold')
    ax.plot(model.daterange, model.immunity('omicron', to_hosp=True, age='65+'), label='Immunity vs Severe Omicron (65+ only)', color='darkorange')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Percent Immune')

    # formatting
    for ax in axs.flatten():
        format_date_axis(ax)
        ax.set_xlim(from_date, to_date)
        ax.axvline(x=dt.date.today(), color='darkgray')
        ax.grid(color='lightgray')
        ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(get_filepath_prefix(outdir) + 'report.png')


def do_build_legacy_output_df(model: CovidModel):
    ydf = model.solution_sum_df(['seir', 'age', 'region']).stack(level='age')
    df = ydf.unstack(level=1).stack(level=1)

    alpha_df = model.get_param_for_attrs_by_t('alpha', attrs={})
    alpha_df['date'] = model.daterange
    alpha_df = alpha_df.reset_index('t').set_index('date', append=True).drop(columns='t')
    combined = model.solution_sum_df()[['E']].stack(model.param_attr_names).join(alpha_df)
    combined['Einc'] = (combined['E'] / combined['alpha'])
    combined = combined.groupby(['date', 'region']).sum()

    totals = model.solution_sum_df(['seir', 'region']).stack(level=1)
    totals = totals.rename(columns={'Ih': 'Iht', 'D': 'Dt', 'E': 'Etotal'})
    totals['Itotal'] = totals['I'] + totals['A']

    #totals_by_priorinf = model.solution_sum(['seir', 'priorinf']) # TODO: update

    #df['Rt'] = totals_by_priorinf[('S', 'none')]  # TODO: update
    #df['Itotal'] = totals['I'] + totals['A']
    #df['Etotal'] = totals['E']
    #df['Einc'] = (combined['E'] / combined['alpha']).groupby('t').sum()
    df.join(totals).join(combined)

    # TODO: how to generalize this for all variants?
    # TODO:  immunity per region?
    df['Vt'] = model.immunity(variant='omicron', vacc_only=True)
    df['immune'] = model.immunity(variant='omicron')
    # why is this needed?
    df['Ilag'] = df['I'].shift(3)

    #df['Re'] = model.re_estimates   # TODO: needs updating in model class
    # update to work with region pop
    #df['prev'] = 100000.0 * df['Itotal'] / model.model_params['total_pop']
    #df['oneinX'] = model.model_params['total_pop'] / df['Itotal']
    # TODO: what is this supposed to be?
    #df['Exposed'] = 100.0 * df['Einc'].cumsum()

    df.index.names = ['t']
    return df

def do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess = None):
    # construct model args from base model args and scenario args list
    model_args_list = []
    for scenario_args in scenario_args_list:
        model_args_list.append(copy.deepcopy(base_model_args))
        model_args_list[-1].update(scenario_args)

    return do_multiple_fits(model_args_list, fit_args, multiprocess)
