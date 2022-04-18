### Python Standard Library ###
import os
from multiprocessing import Pool
### Third Party Imports ###
import matplotlib.pyplot as plt
### Local Imports ###
from covid_model.utils import get_file_prefix
from covid_model.model_specs import CovidModelSpecifications
from covid_model.db import db_engine
from covid_model.model_fit import CovidModelFit
from covid_model.cli_specs import ModelSpecsArgumentParser


# function that runs the fit on each region, can be run via multiprocessing
def do_fit_on_region(args):
    region, specs_args, non_specs_args, perform_specs_write = args
    specs_args['regions'] = [region]

    # initialize fit object
    engine = db_engine()
    fit = CovidModelFit(engine=engine, tc_min=non_specs_args['tc_min'], tc_max=non_specs_args['tc_max'], **specs_args)
    fit.set_actual_hosp(engine=engine, county_ids=fit.base_specs.get_all_county_fips())

    # run fit
    fit.run(engine, **non_specs_args, print_prefix=f'{region}')
    print(fit.fitted_model.tslices)
    print(fit.fitted_tc)

    if non_specs_args['outdir']:
        print("plotting")
        hosps_df = fit.modeled_vs_actual_hosps().reset_index('region').drop(columns='region')
        hosps_df.plot()
        plt.savefig(f"{get_file_prefix(non_specs_args['outdir'])}fit_hosps.png", dpi=300)

        print("saving results")
        hosps_df.to_csv(f"{get_file_prefix(non_specs_args['outdir'])}fit_hospitalized.csv")

    fit.fitted_model.tags['run_type'] = 'fit'
    return fit.fitted_model.write_specs_to_db(engine=engine) if perform_specs_write else fit.fitted_model.prepare_write_specs_query()


def run_fit(look_back, batch_size, increment_size, window_size, tc_min, tc_max, use_base_specs_end_date,
            write_batch_output, multiprocess, forward_sim_each_batch, outdir, **specs_args):
    if(outdir):
        os.makedirs(outdir, exist_ok=True)

    non_specs_args = {}
    for var in ['look_back', 'batch_size', 'increment_size', 'window_size', 'tc_min', 'tc_max', 'use_base_specs_end_date', 'write_batch_output', 'multiprocess', 'forward_sim_each_batch', 'outdir']:
        non_specs_args[var] = eval(var)

    # regions can only be fit individually, so loop through regions if multiple are listed
    if multiprocess:
        args_list = map(lambda x: [x, specs_args, non_specs_args, False], specs_args['regions'])
        p = Pool(multiprocess)
        write_infos = p.map(do_fit_on_region, args_list)

        # write results to database serially
        for i, _ in enumerate(write_infos):
            if write_infos[i] is not None:
                write_infos[i] = CovidModelSpecifications.write_prepared_specs_to_db(write_infos[i], db_engine())
    else:
        args_list = map(lambda x: [x, specs_args, non_specs_args, True], specs_args['regions'])
        write_infos = map(do_fit_on_region, args_list)

    return write_infos


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

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
    parser.set_defaults(refresh_vacc=False, increment_size=1, window_size=14, batch_size=1)

    specs_args = parser.specs_args_as_dict()
    non_specs_args = parser.non_specs_args_as_dict()
    non_specs_args['batch_size'] = non_specs_args['batch_size'] if 'batch_size' in non_specs_args.keys() else non_specs_args['look_back']

    run_fit(**non_specs_args, outdir=outdir, **specs_args)
