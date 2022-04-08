from covid_model.cli_specs import ModelSpecsArgumentParser
from covid_model.model import CovidModel
from covid_model.db import db_engine
from covid_model.data_imports import ExternalHosps

import datetime as dt


if __name__ == '__main__':
    parser = ModelSpecsArgumentParser()
    parser.add_argument('-pd', '--projection_days', type=int, help='number of days to project into the future for retrospective evaluation')
    run_args = parser.parse_args()

    engine = db_engine()
    base_model = CovidModel(engine=engine, **parser.specs_args_as_dict())
    base_model.prep()

    actual_hosps = ExternalHosps(engine, t0_date=base_model.start_date).fetch(county_ids=None)['currently_hospitalized']

    for tc, tslice in zip(base_model.tc, base_model.tslices):
        # shorten model to end tslice + projection_days
        model = CovidModel(base_model=base_model, end_date=base_model.start_date + dt.timedelta(days=tslice + run_args.projection_days))

        # set tc after tslice to the tc value immediately before tslice
        projected_tc = [tc] * len([ts for ts in base_model.tslices if ts >= tslice])
        model.apply_tc(tc=projected_tc)

        # run model
        model.solve_seir()
        print(tslice)
        print(actual_hosps)
        print(model.solution_sum('seir')['Ih'])





