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


def run():
    # get fit params
    parser = argparse.ArgumentParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (default 14-day) windows to look back and refit; default to 3")
    # parser.add_argument("-lbd", "--look_back_date", type=str, help="the date (YYYY-MM-DD) from which we want to refit; default to using -lb, which defaults to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (default 14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-is", "--increment_size", type=int, help="the number of windows to shift forward for each subsequent fit; default to 1")
    parser.add_argument("-ws", "--window_size", type=int, help="the number of days in each TC-window; default to 14")
    parser.add_argument("-f", "--fit_id", type=int, help="the fit_id for the last production fit, which will be used to set historical TC values for windows that will not be refit")
    parser.add_argument("-p", "--params", type=str, help="the path to the params file to use for fitting; default to 'input/params.json'")
    # parser.add_argument("-rv", "--refresh_vacc", type=bool, help="1 if you want to pull new vacc. data from the database, otherwise 0; default 0")
    parser.add_argument("-ahs", "--actual_hosp_sql", type=str, help="path for file containing sql query that fetches actual hospitalization data")
    parser.add_argument("-rv", "--refresh_vacc", action="store_true", help="1 if you want to pull new vacc. data from the database, otherwise 0; default 0")
    parser.add_argument("-om", "--model_with_omicron", action="store_true")
    parser.set_defaults(refresh_vacc=False, model_with_omicron=False)
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    # look_back_date = dt.datetime.strptime(fit_params.look_back_date, '%Y-%m-%d') if fit_params.look_back_date else None
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    increment_size = fit_params.increment_size if fit_params.increment_size is not None else 1
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    fit_id = fit_params.fit_id if fit_params.fit_id is not None else 865
    params = fit_params.params if fit_params.params is not None else 'input/params.json'
    # refresh_vacc_data = fit_params.refresh_vacc if fit_params.refresh_vacc is not None else False
    refresh_vacc_data = fit_params.refresh_vacc
    actual_hosp_sql = fit_params.actual_hosp_sql if fit_params.actual_hosp_sql is not None else 'sql/emresource_hospitalizations.sql'

    # run fit
    engine = db_engine()

    fit = CovidModelFit(fit_id, engine=engine)
    # if params:
    #     fit.base_specs.set_model_params(params)
    # if refresh_vacc_data:
    #     fit.base_specs.set_actual_vacc(engine)

    fit.set_actual_hosp(engine, actual_hosp_sql=actual_hosp_sql)

    fit.run(engine, look_back=look_back, batch_size=batch_size, increment_size=increment_size, window_size=window_size,
                            params='input/params.json',
                            refresh_actual_vacc=refresh_vacc_data,
                            vacc_proj_params=json.load(open('input/vacc_proj_params.json'))['current trajectory'],
                            vacc_immun_params='input/vacc_immun_params.json',
                            param_multipliers='input/param_multipliers.json',
                            variant_prevalence='input/variant_prevalence.csv',
                            mab_prevalence='input/mab_prevalence.csv',
                            model_class=CovidModelWithVariants if fit_params.model_with_omicron else CovidModel)

    # fit.fitted_specs.write_to_db(engine)
    print(fit.fitted_model.specifications.tslices)
    print(fit.fitted_tc)

    fit.fitted_model.specifications.tags['run_type'] = 'fit'
    if fit_params.model_with_omicron:
        fit.fitted_model.specifications.tags['model_type'] = 'with omicron'

    fit.fitted_model.write_to_db(engine)

    actual_hosps(engine)
    modeled(fit.fitted_model, 'Ih')
    plt.show()


if __name__ == '__main__':
    run()
