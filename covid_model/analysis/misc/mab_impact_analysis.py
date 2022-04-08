### Python Standard Library ###
### Third Party Imports ###
import pandas as pd
import matplotlib.pyplot as plt
### Local Imports ###
from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.analysis.charts import modeled, actual_hosps
from covid_model.analysis.misc.vacc_scen_comparison import build_mab_prevalence

if __name__ == '__main__':
    engine = db_engine()

    fig, ax = plt.subplots()
    model = CovidModel([0, 800], engine=engine)
    model.set_ef_from_db(1618)

    base_mab_prevalence = pd.read_csv('input/mab_prevalence.csv', parse_dates=['date'], index_col=0)
    mab_prevalence_target_date = '2021-11-30'

    for mab_uptake in (0.15, 0.30, 0.50):

        model.prep(params='input/params.json', mab_prevalence=build_mab_prevalence(base_mab_prevalence, mab_uptake*0.53, mab_prevalence_target_date))
        model.solve_seir()
        modeled(model, 'Ih')

    actual_hosps(engine)
    plt.show()
