import json

from db import db_engine
from model import CovidModel
from model_fit import CovidModelFit
from analysis.charts import actual_hosps, modeled
import matplotlib.pyplot as plt
from covid_model.cli_specs import ModelSpecsCliParser


def run():
    # get fit params
    parser = ModelSpecsCliParser()
    parser.add_argument("-lb", "--look_back", type=int, help="the number of (default 14-day) windows to look back and refit; default to 3")
    parser.add_argument("-bs", "--batch_size", type=int, help="the number of (default 14-day) windows to fit in each batch; default to running everything in one batch")
    parser.add_argument("-is", "--increment_size", type=int, help="the number of windows to shift forward for each subsequent fit; default to 1")
    parser.add_argument("-ws", "--window_size", type=int, help="the number of days in each TC-window; default to 14")
    parser.add_argument("-ahs", "--actual_hosp_sql", type=str, help="path for file containing sql query that fetches actual hospitalization data")
    parser.set_defaults(refresh_vacc=False)
    fit_params = parser.parse_args()
    look_back = fit_params.look_back
    batch_size = fit_params.batch_size if fit_params.batch_size is not None else look_back
    increment_size = fit_params.increment_size if fit_params.increment_size is not None else 1
    window_size = fit_params.window_size if fit_params.window_size is not None else 14
    actual_hosp_sql = fit_params.actual_hosp_sql if fit_params.actual_hosp_sql is not None else 'sql/emresource_hospitalizations.sql'

    # run fit
    engine = db_engine()

    fit = CovidModelFit(engine=engine, **parser.specs_args_as_dict())
    fit.set_actual_hosp(engine, actual_hosp_sql=actual_hosp_sql)
    fit.run(engine, look_back=look_back, batch_size=batch_size, increment_size=increment_size, window_size=window_size)

    print(fit.fitted_model.specifications.tslices)
    print(fit.fitted_tc)

    fit.fitted_model.specifications.tags['run_type'] = 'fit'
    fit.fitted_model.specifications.write_to_db(engine)

    actual_hosps(engine)
    modeled(fit.fitted_model, 'Ih')
    plt.show()


if __name__ == '__main__':
    run()
