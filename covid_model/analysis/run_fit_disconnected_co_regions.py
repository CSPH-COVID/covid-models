import matplotlib.pyplot as plt
import json
import datetime as dt
from multiprocessing import Pool

from covid_model.db import db_engine
from covid_model import RegionalCovidModel, all_regions
from covid_model.model_specs import CovidModelSpecifications
from covid_model.model_fit import CovidModelFit
from covid_model.analysis.charts import actual_hosps, modeled
from covid_model.cli_specs import ModelSpecsArgumentParser
from covid_model.data_imports import ExternalHosps


def region_fit(args):
    region, region_params, fit_params, specs_args, look_back, batch_size, increment_size, window_size, write_batch_output = args
    rname = all_regions[region]
    print(f'Region: {rname}')
    try:
        engine = db_engine()
        region_county_ids = json.load(open(region_params))[region]['county_fips']
        print(f'{rname}: Creating fit object and loading hospitalizations')
        fit = CovidModelFit(engine=engine, region_params=fit_params.region_params, region=region,
                            tc_min=fit_params.tc_min, tc_max=fit_params.tc_max, **specs_args)
        fit.set_actual_hosp(engine, county_ids=region_county_ids)

        # run fit
        print(f'{rname}: Fitting')
        fit.run(engine, look_back=look_back, batch_size=batch_size,
                increment_size=increment_size, window_size=window_size,
                region_params=region_params, region=region,
                write_batch_output=write_batch_output,
                model_class=RegionalCovidModel,
                model_args={"region": region},
                forward_sim_each_batch=True, use_base_specs_end_date=True)

        fit.fitted_model.tags['run_type'] = 'fit'

        if fit_params.plot:
            print(f'{rname}: Plotting')
            actual_hosps(engine, county_ids=region_county_ids)
            modeled(fit.fitted_model, 'Ih')
            plt.savefig('output/fitted_' + region + ".png", dpi=300)
        print(f'{rname}: Succeeded, spec_id:{fit.fitted_model.spec_id}')
        return fit.fitted_model.preform_write_query()
    except RuntimeError as e:
        print(e)
        print(f'{rname}: Failed')
        return None


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (default 14-day) windows to look back and refit; defaults to refitting all windows")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (default 14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-is", "--increment_size", type=int, help="the number of windows to shift forward for each subsequent fit; default to 1")
    parser.add_argument("-ws", "--window_size", type=int, help="the number of days in each TC-window; default to 14")
    parser.add_argument("-tmn", "--tc_min", type=float, default=0, help="The lowest tc to allow")
    parser.add_argument("-tmx", "--tc_max", type=float, default=0.99, help="The lowest tc to allow")
    parser.add_argument("-rp", "--region_params", type=str, default="input/region_params.json", help="the path to the region-specific params file to use for fitting; default to 'input/region_params.json'")
    parser.add_argument("-rg", "--regions", nargs="+", choices=all_regions.keys(), required=False, help="Specify the regions to be run, default is all regions (not counties)")
    parser.add_argument("-wb", "--write_batch_output", action="store_true", default=False, help="write the output of each batch to the database")
    parser.add_argument("-plt", "--plot", action="store_true", default=False, help="open a plot when fitting is complete, comparing modeled hosp to actual; default False")
    parser.add_argument("-mp", "--multiprocess", default=None, help="if present, indicates how many indipendent processes to spawn in a threadpool for fitting the models")
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    increment_size = fit_params.increment_size if fit_params.increment_size is not None else 1
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    regions = fit_params.regions
    region_params = fit_params.region_params
    write_batch_output = fit_params.write_batch_output
    multiprocess = int(fit_params.multiprocess) if fit_params.multiprocess else None
    specs_args = parser.specs_args_as_dict()

    args = [region_params, fit_params, specs_args, look_back, batch_size, increment_size, window_size, write_batch_output]

    if multiprocess:
        p = Pool(multiprocess)
        write_infos = p.map(region_fit, map(lambda x: [x] + args, regions))
    else:
        write_infos = map(region_fit, map(lambda x: [x] + args, regions))

    for write_info in write_infos:
        if write_info is not None:
            CovidModelSpecifications.write_preformed_to_db(write_info, db_engine())


if __name__ == '__main__':
    run()
