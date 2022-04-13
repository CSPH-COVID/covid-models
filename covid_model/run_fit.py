### Python Standard Library ###
from multiprocessing import Pool
### Third Party Imports ###
import matplotlib.pyplot as plt
import pandas as pd
### Local Imports ###
from covid_model.model_specs import CovidModelSpecifications
from covid_model.db import db_engine
from covid_model.model_fit import CovidModelFit
from covid_model.analysis.charts import actual_hosps, modeled
from covid_model.cli_specs import ModelSpecsArgumentParser


def do_fit_on_region(args):
    region, specs_args, non_specs_args, perform_specs_write = args

    # initialize fit object
    engine = db_engine()
    fit = CovidModelFit(engine=engine, tc_min=non_specs_args['tc_min'], tc_max=non_specs_args['tc_max'], **specs_args)
    fit.set_actual_hosp(engine=engine, county_ids=fit.base_specs.get_all_county_fips())


    # run fit
    fit.run(engine, **non_specs_args, print_prefix=f'{region}:')
    print(fit.fitted_model.tslices)
    print(fit.fitted_tc)

    if non_specs_args['plot']:
        # TODO: update
        # actual_hosps(engine, county_ids=)
        modeled(fit.fitted_model, 'Ih')
        plt.show()

    fit.fitted_model.tags['run_type'] = 'fit'
    return fit.fitted_model.write_specs_to_db(engine=engine) if perform_specs_write else fit.fitted_model.prepare_write_specs_query()


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (default 14-day) windows to look back and refit; defaults to refitting all windows")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (default 14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-is", "--increment_size", type=int, help="the number of windows to shift forward for each subsequent fit; default to 1")
    parser.add_argument("-ws", "--window_size", type=int, help="the number of days in each TC-window; default to 14")
    parser.add_argument("-tmn", "--tc_min", type=float, default=0, help="The lowest tc to allow")
    parser.add_argument("-tmx", "--tc_max", type=float, default=0.99, help="The lowest tc to allow")
    parser.add_argument("-bsed", '--use_base_specs_end_date', action="store_true", default=False, help="use end date from model specifications, not hospitalizations data")
    parser.add_argument("-wb", "--write_batch_output", action="store_true", default=False, help="write the output of each batch to the database")
    parser.add_argument("-mup", "--multiprocess", type=int, default=None, help="if present, indicates how many indipendent processes to spawn in a threadpool for fitting the models")
    parser.add_argument("-fs", "--forward_sim_each_batch", action='store_true', default=False, help="after each tc fit, plot the hospitalized vs. fitted, along with TC")
    parser.add_argument("-plt", "--plot", action="store_true", default=False, help="open a plot when fitting is complete, comparing modeled hosp to actual; default False")
    parser.set_defaults(refresh_vacc=False, increment_size=1, window_size=14, batch_size=1)

    fit_params = parser.parse_args()
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else fit_params.look_back

    # regions can only be fit individually, so loop through regions if multiple are listed
    regions = fit_params.regions
    if fit_params.multiprocess:
        args_list = map(lambda x: [x, parser.specs_args_as_dict(update={'regions': [x]}), parser.non_specs_args_as_dict(update={"batch_size": batch_size}), False], regions)
        p = Pool(fit_params.multiprocess)
        write_infos = p.map(do_fit_on_region, args_list)

        # write results to database serially
        for write_info in write_infos:
            if write_info is not None:
                CovidModelSpecifications.write_prepared_specs_to_db(write_info, db_engine())
    else:
        args_list = map(lambda x: [x, parser.specs_args_as_dict(update={'regions': [x]}), parser.non_specs_args_as_dict(update={"batch_size": batch_size}), True], regions)
        results = map(do_fit_on_region, args_list)
        any(results) # force execution of the map


if __name__ == '__main__':
    run()
