### Python Standard Library ###
import datetime as dt
### Third Party Imports ###
from matplotlib import pyplot as plt
### Local Imports ###
from covid_model.analysis.misc.dynamic_projections import get_tc_sims
from covid_model.model_fit import CovidModelFit
from db_utils.conn import db_engine

if __name__ == '__main__':
    engine = db_engine()
    fit = CovidModelFit.from_db(engine, 1865)


    get_tc_sims(fit, sample_count=500, samples_per_fit_sim=10, max_date=dt.datetime(2022, 5, 31), plot=True, arima_order=(1, 0, 0), skip=8)
    # get_tc_sims(fit, sample_count=500, samples_per_fit_sim=10, max_date=dt.datetime(2022, 5, 31), plot=True, arima_order=(2, 0, 1), skip=8)
    # get_tc_sims(fit, sample_count=300, samples_per_fit_sim=10, max_date=dt.datetime(2022, 5, 31), plot=True, arima_order='auto', skip=8)
    # get_tc_sims(fit, sample_count=500, samples_per_fit_sim=10, max_date=dt.datetime(2022, 5, 31), plot=True, arima_order=(2, 0, 1), skip=8)
    # get_tc_sims(fit, sample_count=500, samples_per_fit_sim=10, max_date=dt.datetime(2022, 5, 31), plot=True, arima_order='auto', skip=8)
    plt.show()
