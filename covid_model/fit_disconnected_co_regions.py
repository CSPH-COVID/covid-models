import matplotlib.pyplot as plt
import json
import datetime as dt

from db import db_engine
from covid_model import RegionalCovidModel, all_regions
from model_fit import CovidModelFit
from analysis.charts import actual_hosps, modeled
from covid_model.cli_specs import ModelSpecsArgumentParser
from data_imports import ExternalHosps


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
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    increment_size = fit_params.increment_size if fit_params.increment_size is not None else 1
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    regions = fit_params.regions
    region_params = fit_params.region_params
    write_batch_output = fit_params.write_batch_output

    engine = db_engine()

    # run fit
    for region in regions:
        print(f'Region: {all_regions[region]}')
        try:
            # get hospitalization data and adjust start/end dates
            print("Adjusting start / end dates")
            region_counties = json.load(open(region_params))[region]['county_names']
            region_county_ids = json.load(open(region_params))[region]['county_fips']
            hosps = ExternalHosps(db_engine()).fetch_from_db(region_county_ids)
            fit_params.start_date = max(dt.datetime.strptime(fit_params.start_date, "%Y-%m-%d").date(),
                                        min(hosps.index[hosps['currently_hospitalized'] > 0]) - dt.timedelta(days=1)).strftime("%Y-%m-%d")
            fit_params.end_date = min(dt.datetime.strptime(fit_params.end_date, "%Y-%m-%d").date(),
                                        max(hosps.index[hosps['currently_hospitalized'] > 0])).strftime("%Y-%m-%d")
            print(f'Adjusted start and end dates: {fit_params.start_date}, {fit_params.end_date}')
            print('Creating fit object and loading hospitalizations')
            fit = CovidModelFit(engine=engine, region_params=fit_params.region_params, region=region, tc_min=fit_params.tc_min, tc_max=fit_params.tc_max, **parser.specs_args_as_dict())
            fit.set_actual_hosp(engine, county_ids=region_county_ids)

            # run fit
            print('Fitting')
            fit.run(engine, look_back=look_back, batch_size=batch_size,
                    increment_size=increment_size, window_size=window_size,
                    region_params=region_params, region=region,
                    write_batch_output=write_batch_output,
                    model_class=RegionalCovidModel,
                    model_args={"region": region} if regions is not None else dict(),
                    forward_sim_each_batch=True)

            fit.fitted_model.tags['run_type'] = 'fit'
            fit.fitted_model.write_to_db(engine)
            if fit_params.plot:
                print('Plotting')
                actual_hosps(engine, county_ids=region_county_ids)
                modeled(fit.fitted_model, 'Ih')
                plt.savefig('output/fitted_' + region + ".png", dpi=300)
            print(f'{region} succeeded')
        except RuntimeError:
            print(f'{region} failed')


if __name__ == '__main__':
    run()
