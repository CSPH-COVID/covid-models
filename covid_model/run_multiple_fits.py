### Python Standard Library ###
import json
import datetime as dt
### Third Party Imports ###
import matplotlib.pyplot as plt
import pandas as pd
### Local Imports ###
from db import db_engine
from model_fit import CovidModelFit
from analysis.charts import actual_hosps, modeled
from covid_model.cli_specs import ModelSpecsArgumentParser

regions = {
  "ad": "Adams County",
  "ar": "Arapahoe County",
  "bo": "Boulder County",
  "brm": "Broomfield County",
  "den": "Denver County",
  "doug": "Douglas County",
  "ep": "El Paso County",
  "jeff": "Jefferson County",
  "lar": "Larimer County",
  "mesa": "Mesa County",
  "pueb": "Pueblo County",
  "weld": "Weld County",
  "cent": "Central",
  "cm": "Central Mountains",
  "met": "Metro",
  "ms": "Metro South",
  "ne": "Northeast",
  "nw": "Northwest",
  "slv": "San Luis Valley",
  "sc": "South Central",
  "sec": "Southeast Central",
  "sw": "Southwest",
  "wcp": "West Central Partnership"
}


def run():
    # get fit params
    parser = ModelSpecsArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (default 14-day) windows to look back and refit; defaults to refitting all windows")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (default 14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-is", "--increment_size", type=int, help="the number of windows to shift forward for each subsequent fit; default to 1")
    parser.add_argument("-ws", "--window_size", type=int, help="the number of days in each TC-window; default to 14")
    parser.add_argument("-rp", "--region_params", type=str, default="input/region_params.json", help="the path to the region-specific params file to use for fitting; default to 'input/region_params.json'")
    parser.add_argument("-rg", "--region", choices=regions.keys(), required=False, help="Specify the region to be run, if not specified, just runs default parameters")
    parser.add_argument("-rh", "--hosp_data", type=str, help="the path to the hospitalizations data for regions (temporary fix)")
    parser.add_argument("-wb", "--write_batch_output", action="store_true", default=False, help="write the output of each batch to the database")
    parser.add_argument("-plt", "--plot", action="store_true", default=False, help="open a plot when fitting is complete, comparing modeled hosp to actual; default False")
    parser.add_argument("-ambs", "--attribute_multipliers_by_scen", type=str, help="file containing attribute multipliers, broken out by scenario")
    parser.add_argument("-ct", "--context_tag", type=str, help="a tag that will be added to the specification as the \"context\"")
    parser.set_defaults(refresh_vacc=False)
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    increment_size = fit_params.increment_size if fit_params.increment_size is not None else 1
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    region_params = fit_params.region_params
    region = fit_params.region
    write_batch_output = fit_params.write_batch_output

    # run fit
    engine = db_engine()

    with open(fit_params.attribute_multipliers_by_scen) as f:
        attr_mults_by_scen = json.load(f)

    common_attr_mults = attr_mults_by_scen['common']
    for scen, scen_attr_mults in attr_mults_by_scen['scens'].items():
        print(f'Running fit for scenario: "{scen}"')
        attr_mults = common_attr_mults + scen_attr_mults
        specs_args = {**parser.specs_args_as_dict(), 'attribute_multipliers': attr_mults}
        fit = CovidModelFit(engine=engine, region_params=fit_params.region_params, region=fit_params.region, **specs_args)

        # extract county IDs for the provided region; if statewide, set counties=None
        if region is not None:
            counties = json.load(open(region_params))[region]['county_names']
            counties = counties if type(counties) == list else [counties]
            county_ids = json.load(open(region_params))[region]['county_fips']
            county_ids = county_ids if type(county_ids) == list else [county_ids]
        else:
            counties, county_ids = (None, None)

        # if hospitalization data CSV is provided, get hospitalizations from there; else query the database
        if fit_params.hosp_data is not None:
            if region is not None:
                hosp_data = pd.read_csv(fit_params.hosp_data)[['date'] + counties]
                tstart = pd.to_datetime(hosp_data['date']).min()
                fit.base_specs.start_date = dt.date(tstart.year, tstart.month, tstart.day)
                fit.actual_hosp = hosp_data.drop('date', axis=1).sum(axis=1)
            else:
                raise ValueError('Hospitalization data file is not supported for statewide model fitting.')
        else:
            fit.set_actual_hosp(engine, county_ids=county_ids)

        # run fit
        if look_back == 0:
            fit.fitted_model = fit.base_specs
        else:
            fit.run(engine, look_back=look_back, batch_size=batch_size,
                    increment_size=increment_size, window_size=window_size,
                    region_params=region_params, region=region,
                    write_batch_output=write_batch_output)

        print(fit.fitted_model.tslices)
        print(fit.fitted_tc)

        fit.fitted_model.tags['run_type'] = 'scenario fit'
        fit.fitted_model.tags['context'] = fit_params.context_tag
        fit.fitted_model.tags['scenario'] = scen
        fit.fitted_model.write_to_db(engine)

        if fit_params.plot:
            actual_hosps(engine)
            modeled(fit.fitted_model, 'Ih')
            plt.draw()

    plt.show()


if __name__ == '__main__':
    run()
