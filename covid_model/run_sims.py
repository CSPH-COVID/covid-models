### Python Standard Library ###
import json
import datetime as dt
import argparse
### Third Party Imports ###
import numpy as np
### Local Imports ###
from covid_model.db import db_engine
from covid_model.model_specs import CovidModelSpecifications
from covid_model.model_sims import CovidModelSimulation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sid", "--spec_id", type=int, help="the base spec ID")
    parser.add_argument("-n", "--number_of_sims", type=int, help="number of simulations to run")
    parser.add_argument("-mab", "--mab_uptake_target", type=float, help="the share of eligible who will be getting mAbs in 4 weeks")
    parser.add_argument("-vp", "--vacc_proj", type=str, help="the name of the vaccine projection to use")
    run_params = parser.parse_args()

    engine = db_engine()

    end_date = dt.date(2022, 5, 31)

    specs = CovidModelSpecifications(engine=engine, spec_id=run_params.spec_id, new_end_date=end_date)
    specs.spec_id = None
    specs.tags['run_type'] = 'sim'

    if run_params.mab_uptake_target is not None:
        days_to_hit_target = 28
        target_prevalence = np.round(0.53 * run_params.mab_uptake_target, 4)
        mab_params = specs.timeseries_effects['mab'][0]
        if len(mab_params['start_date']) == 8:
            mab_params['start_date'] = '20' + mab_params['start_date']
        todays_index = (dt.date.today() - dt.datetime.strptime(mab_params['start_date'], '%Y-%m-%d').date()).days
        current_prevalence = mab_params['prevalence'][todays_index]
        mab_params['prevalence'][todays_index:todays_index + days_to_hit_target] = np.round(np.linspace(current_prevalence, target_prevalence, days_to_hit_target), 4)
        mab_params['prevalence'][todays_index + days_to_hit_target:] = [target_prevalence] * (len(mab_params['prevalence']) - (todays_index + days_to_hit_target))
        specs.tags['mab_target'] = run_params.mab_uptake_target
        print(mab_params['prevalence'])

    if run_params.vacc_proj is not None:
        specs.set_vacc_proj(json.load(open('input/vacc_proj_params.json'))[run_params.vacc_proj])
        specs.tags['vacc_proj'] = run_params.vacc_proj

    specs.write_to_db(engine)
    sims = CovidModelSimulation(specs, engine=engine, end_date=end_date)
    sims.run_base_result()
    sims.run_simulations(run_params.number_of_sims, sims_per_fitted_sample=10)
