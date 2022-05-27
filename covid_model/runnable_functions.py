### Python Standard Library ###
import os
from multiprocessing import Pool
import logging
import json
from time import perf_counter
import datetime as dt
### Third Party Imports ###
from multiprocessing_logging import install_mp_handler
from scipy import optimize as spo
from matplotlib import pyplot as plt
### Local Imports ###
from covid_model import CovidModel, db_engine
from covid_model.analysis.charts import plot_transmission_control
from covid_model.utils import IndentLogger, setup, get_file_prefix
logger = IndentLogger(logging.getLogger(''), {})


def __single_window_fit(model: CovidModel, look_back, tc_min, tc_max, y0d=None):
    # define initial states
    fixed_tc = model.tc[:-look_back]
    y0d = model.y0_dict if y0d is None else y0d
    y0 = model.y0_from_dict(y0d)

    # function to be optimized
    def func(trange, *test_tc):
        combined_tc = fixed_tc + list(test_tc)
        model.apply_tc(combined_tc)
        model.solve_seir(y0=y0, method='RK45')
        return model.solution_sum_Ih()
    fitted_tc, fitted_tc_cov = spo.curve_fit(
        f=func
        , xdata=model.trange
        , ydata=model.actual_hosp[:len(model.trange)]
        , p0=model.tc[-look_back:]
        , bounds=([tc_min] * look_back, [tc_max] * look_back))
    return fitted_tc, fitted_tc_cov


def do_single_fit(tc_0=0.75,  # default value for TC
                  tc_min=0,  # minimum allowable TC
                  tc_max=0.99,  # maximum allowable TC
                  window_size=14,  # How often to update TC (days)
                  look_back=None,  # How many tc values to refit (if None, refit all)
                  last_window_min_size=21,  # smallest size of the last TC window
                  batch_size=None,  # How many windows to fit at once
                  increment_size=1,  # How many windows to shift over for each fit
                  prep_model=True,  # Should we run model.prep
                  use_hosps_end_date=True,  # Should we fit all the way to the end_date of the hospitalization data (as opposed to the model's end date)
                  outdir=None,  # the output directory for saving results
                  write_results=True,  # should final results be written to the database
                  write_batch_results=False,  # Should we write output to the database after each fit
                  **model_args):

    def forward_sim_plot(model):
        # TODO: refactor into charts?
        logger.info(f'{str(model.tags)}: Running forward sim')
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_subplot(211)
        hosps_df = model.modeled_vs_actual_hosps().reset_index('region').drop(columns='region')
        hosps_df.plot(ax=ax)
        ax = fig.add_subplot(212)
        plot_transmission_control(model, ax=ax)
        plt.savefig(get_file_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_fit.png')
        plt.close()
        hosps_df.to_csv(get_file_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_fit.csv')
        json.dump(dict(dict(zip([0] + model.tc_tslices, model.tc))), open(get_file_prefix(outdir) + f'{"_".join(str(key) + "_" + str(val) for key, val in model.tags.items())}_model_tc.json', 'w'))

#    logging.info(json.dumps({"fit_args": fit_args}, default=str))
    logging.info(str({"model_build_args": model_args}))

    # initialize and run the fit
    if write_batch_results or write_results:
        engine = db_engine()

    base_model = CovidModel(**model_args)

    # get the end date from actual hosps
    if use_hosps_end_date:
        base_model.end_date = base_model.actual_hosp.index.max()
    elif base_model.actual_hosp.index.max() < base_model.end_date:
        ermsg = f'Opted to fit to model end_date, {base_model.end_date}, but hospitalizations only available up to {base_model.actual_hosp.index.max()}'
        logger.exception(f"{str(base_model.tags)}" + ermsg)
        raise ValueError(ermsg)

    # prep model (we only do this once to save time)
    if prep_model:
        logger.info(f'{str(base_model.tags)} Prepping Model')
        t0 = perf_counter()
        base_model.prep()
        logger.info(f'{str(base_model.tags)} Model prepped for fitting in {perf_counter() - t0} seconds.')

    # Apply TC
    tslices = base_model.tc_tslices  + list(range((base_model.tc_tslices[-1] if len(base_model.tc_tslices) > 0 else 0) + window_size, base_model.tmax - last_window_min_size, window_size))
    tc = base_model.tc + [tc_0] * (len(tslices) + 1 - len(base_model.tc))
    base_model.apply_tc(tcs=tc, tslices=tslices)

    # run fit
    fitted_tc_cov = None
    if look_back is None:
        look_back = len(tslices) + 1

    # if there's no batch size, set the batch size to be the total number of windows to be fit
    if batch_size is None or batch_size > look_back:
        batch_size = look_back

    trim_off_end_list = list(range(look_back - batch_size, 0, -increment_size)) + [0]
    logger.info(f'{str(base_model.tags)} Will fit {len(trim_off_end_list)} times')
    for i, trim_off_end in enumerate(trim_off_end_list):
        t0 = perf_counter()
        this_end_t = tslices[-trim_off_end] if trim_off_end > 0 else base_model.tmax
        this_end_date = base_model.start_date + dt.timedelta(days=this_end_t)

        t01 = perf_counter()
        model = CovidModel(base_model=base_model, end_date=this_end_date, update_derived_properties=False)
        model.apply_tc(tc[:len(tc) - trim_off_end], tslices=tslices[:len(tslices) - trim_off_end])
        logger.info(f'{str(model.tags)}: Model copied in {perf_counter() - t01} seconds.')

        fitted_tc, fitted_tc_cov = __single_window_fit(model, look_back=batch_size, tc_min=tc_min, tc_max=tc_max)
        tc[len(tc) - trim_off_end - batch_size:len(tc) - trim_off_end] = fitted_tc
        logger.info(f'{str(model.tags)}: Transmission control fit {i + 1}/{len(trim_off_end_list)} completed in {perf_counter() - t0} seconds: {fitted_tc}')
        model.tags['fit_batch'] = str(i)

        if write_batch_results:
            logger.info(f'{str(model.tags)}: Uploading batch results')
            model.write_specs_to_db(engine)
            logger.info(f'{str(model.tags)}: spec_id: {model.spec_id}')

        # simulate the model and save a picture of the output
        forward_sim_plot(model)

    model.tc_cov = fitted_tc_cov
    model.tags['run_type'] = 'fit'
    logger.info(f'{str(model.tags)}: tslices: {str(model.tc_tslices)}')
    logger.info(f'{str(model.tags)}: fitted TC: {str(fitted_tc)}')

    if outdir is not None:
        forward_sim_plot(model)

    if write_results:
        logger.info(f'{str(base_model.tags)}: Uploading final results')
        model.write_specs_to_db(engine)
        model.write_results_to_db(engine)

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

