import matplotlib.pyplot as plt
import pandas as pd
import json
import datetime as dt

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
    parser.add_argument("-ahs", "--actual_hosp_sql", type=str, help="path for file containing sql query that fetches actual hospitalization data")
    parser.add_argument("-rp", "--region_params", type=str, default="input/region_params.json", help="the path to the region-specific params file to use for fitting; default to 'input/region_params.json'")
    parser.add_argument("-rg", "--region", choices=regions.keys(), required=False, help="Specify the region to be run, if not specified, just runs default parameters")
    parser.add_argument("-hd", "--hosp_data", type=str, help="the path to the hospitalizations data for regions (temporary fix)")
    parser.add_argument("-wb", "--write_batch_output", action="store_true", default=False, help="write the output of each batch to the database")
    parser.set_defaults(refresh_vacc=False)
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    increment_size = fit_params.increment_size if fit_params.increment_size is not None else 1
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    region_params = fit_params.region_params
    region = fit_params.region
    write_batch_output = fit_params.write_batch_output
    actual_hosp_sql = fit_params.actual_hosp_sql if fit_params.actual_hosp_sql is not None else 'sql/emresource_hospitalizations.sql'

    # run fit
    engine = db_engine()

    fit = CovidModelFit(engine=engine, **parser.specs_args_as_dict())

    ####################################################################################################################
    # temporary code, currently loads region hospitalizations from  local files
    if region is None:
        actual_hosp_sql = fit_params.actual_hosp_sql if fit_params.actual_hosp_sql is not None else 'sql/emresource_hospitalizations.sql'
        fit.set_actual_hosp(engine, actual_hosp_sql=actual_hosp_sql)
    else:
        counties = json.load(open(region_params))[region]['county_names']
        counties = counties if type(counties) == list else [counties]
        hosp_data = pd.read_csv(fit_params.hosp_data)[['date'] + counties]
        tstart = pd.to_datetime(hosp_data['date']).min()
        tend = pd.to_datetime(hosp_data['date']).max()
        fit.base_specs.start_date = dt.date(tstart.year, tstart.month, tstart.day)
        #fit.base_specs.end_date = dt.date(tend.year, tend.month, tend.day)
        fit.actual_hosp = hosp_data.drop('date', axis=1).sum(axis=1)
    ####################################################################################################################

    fit.run(engine, look_back=look_back, batch_size=batch_size,
            increment_size=increment_size, window_size=window_size,
            region_params=region_params, region=region,
            write_batch_output=write_batch_output)

    print(fit.fitted_model.specifications.tslices)
    print(fit.fitted_tc)

    fit.fitted_model.specifications.tags['run_type'] = 'fit'
    fit.fitted_model.specifications.write_to_db(engine)

    #actual_hosps(engine)
    #modeled(fit.fitted_model, 'Ih')
    #plt.show()


if __name__ == '__main__':
    run()
