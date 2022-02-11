import json

from db import db_engine
from model import CovidModel, CovidModelSpecifications
from model_with_omicron import CovidModelWithVariants
from model_fit import CovidModelFit
from data_imports import ExternalHosps
from analysis.charts import actual_hosps, modeled
import matplotlib.pyplot as plt
import datetime as dt
from time import perf_counter
import argparse
import pandas as pd


regions = {
  "ad": "Adams County",
  "ar": "Arapahoe County",
  "bo": "Boulder County",
  "br": "Broomfield County",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (default 14-day) windows to look back and refit; defaults to refitting all windows")
    # parser.add_argument("-lbd", "--look_back_date", type=str, help="the date (YYYY-MM-DD) from which we want to refit; default to using -lb, which defaults to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (default 14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-is", "--increment_size", type=int, help="the number of windows to shift forward for each subsequent fit; default to 1")
    parser.add_argument("-ws", "--window_size", type=int, help="the number of days in each TC-window; default to 14")
    parser.add_argument("-f", "--fit_id", type=int, help="the fit_id for the last production fit, which will be used to set historical TC values for windows that will not be refit")
    parser.add_argument("-p", "--params", type=str, help="the path to the params file to use for fitting; default to 'input/params.json'")
    parser.add_argument("-rp", "--region_params", type=str, default="input/region_params.json", help="the path to the region-specific params file to use for fitting; default to 'input/region_params.json'")
    # parser.add_argument("-rv", "--refresh_vacc", type=bool, help="1 if you want to pull new vacc. data from the database, otherwise 0; default 0")
    parser.add_argument("-ahs", "--actual_hosp_sql", type=str, help="path for file containing sql query that fetches actual hospitalization data")
    parser.add_argument("-rv", "--refresh_vacc", action="store_true", help="1 if you want to pull new vacc. data from the database, otherwise 0; default 0")
    parser.add_argument("-om", "--model_with_omicron", action="store_true")
    parser.add_argument("-rg", "--region", choices=regions.keys(), required=False, help="Specify the region to be run, if not specified, just runs default parameters")
    parser.add_argument("-hd", "--hosp_data", type=str, help="the path to the hospitalizations data for regions (temporary fix)")
    parser.add_argument("-wb", "--write_batch_output", action="store_true", default=False, help="write the output of each batch to the database")
    parser.set_defaults(refresh_vacc=False, model_with_omicron=False)
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    # look_back_date = dt.datetime.strptime(fit_params.look_back_date, '%Y-%m-%d') if fit_params.look_back_date else None
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    increment_size = fit_params.increment_size if fit_params.increment_size is not None else 1
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    fit_id = fit_params.fit_id if fit_params.fit_id is not None else 865
    params = fit_params.params if fit_params.params is not None else 'input/params.json'
    refresh_vacc_data = fit_params.refresh_vacc
    region_params = fit_params.region_params
    region = fit_params.region
    write_batch_output = fit_params.write_batch_output

    # run fit
    engine = db_engine()

    fit = CovidModelFit(fit_id, engine=engine)

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



    fit.run(engine, look_back=look_back, batch_size=batch_size, increment_size=increment_size, window_size=window_size,
            params=params,
            refresh_actual_vacc=refresh_vacc_data,
            vacc_proj_params=json.load(open('input/vacc_proj_params.json'))['current trajectory'],
            vacc_immun_params='input/vacc_immun_params.json',
            timeseries_effect_multipliers='input/timeseries_effects/multipliers.json',
            variant_prevalence='input/timeseries_effects/variant_prevalence.csv',
            mab_prevalence='input/timeseries_effects/mab_prevalence.csv',
            model_class=CovidModelWithVariants if fit_params.model_with_omicron else CovidModel,
            attribute_multipliers='input/attribute_multipliers.json' if fit_params.model_with_omicron else None,
            region_params=region_params,
            region=region,
            write_batch_output=write_batch_output
            )

    # fit.fitted_specs.write_to_db(engine)
    print(fit.fitted_model.specifications.tslices)
    print(fit.fitted_tc)

    fit.fitted_model.specifications.tags['run_type'] = 'fit'
    if fit_params.model_with_omicron:
        fit.fitted_model.specifications.tags['model_type'] = 'with omicron'

    fit.fitted_model.specifications.write_to_db(engine)
    # fit.fitted_model.write_to_db(engine)

    #actual_hosps(engine)
    #modeled(fit.fitted_model, 'Ih')
    #plt.show()


if __name__ == '__main__':
    run()
