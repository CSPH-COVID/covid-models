import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import datetime as dt
import json
from charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates, format_date_axis
from time import perf_counter
from timeit import timeit
import argparse

from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.model_with_future_variant import CovidModelWithFutureVariant
from covid_model.cli_specs import ModelSpecsArgumentParser
# from covid_model.model_with_omicron import CovidModelWithVariants
# from covid_model.model_with_immunity_rework import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications
from covid_model.run_model_scenarios import build_legacy_output_df


if __name__ == '__main__':
    # parser = ModelSpecsArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("-sids", "--spec_ids", type=int, nargs='+', help="specification IDs to compare")
    parser.add_argument("-p", "--params", type=str, help="overwerite model params")
    parser.add_argument("-fd", "--from_date", default='2021-07-01', type=str, help="min date for plots")
    parser.add_argument("-td", "--to_date", default='2022-12-31', type=str, help="max date for plots")
    parser.add_argument("-ftc", "--future_tc", type=float, help="TC to converge to in the coming weeks")
    parser.add_argument("-wtcs", "--winter_tc_shift", type=float, help="TC shift to apply from Nov 25 - Jan 31")
    parser.add_argument("-fvsd", "--future_variant_seed_date", type=str, help="the date at which to seed a future variant")
    parser.add_argument("-fvam", "--future_variant_attribute_multipliers", type=str, help="the attribute multipliers to append for the future variant")
    run_args = parser.parse_args()
    from_date = dt.date.fromisoformat(run_args.from_date)
    to_date = dt.date.fromisoformat(run_args.to_date)
    future_variant_seed_date = dt.date.fromisoformat(run_args.future_variant_seed_date) if run_args.future_variant_seed_date else None

    engine = db_engine()

    fig, ax = plt.subplots()
    actual_hosps(engine, ax=ax, color='black')

    for spec_id in run_args.spec_ids:
        # create models
        models = dict()
        models['no future variant'] = CovidModel(end_date=to_date, engine=engine, from_specs=spec_id, params=run_args.params)
        if future_variant_seed_date is not None:
            models['w/ future variant'] = CovidModelWithFutureVariant(end_date=to_date, engine=engine, from_specs=spec_id, future_seed_date=future_variant_seed_date, params=run_args.params)
            # add attribute multipliers for future variant
            if run_args.future_variant_attribute_multipliers is not None:
                models['w/ future variant'].attribute_multipliers = [am for am in models['w/ future variant'].attribute_multipliers if 'variant' not in am['attrs'].keys() or am['attrs']['variant'] != 'future']  # remove attr mults for future variant
                models['w/ future variant'].attribute_multipliers += run_args.future_variant_attribute_multipliers if isinstance(run_args.future_variant_attribute_multipliers, list) else json.load(open(run_args.future_variant_attribute_multipliers))  # add new attr mults for future variant

        # prep, run, and plot models
        for sublabel, model in models.items():
            t0 = perf_counter()
            print(f'Prepping model for specification {spec_id}...')

            # adjust TC to future TC over the next 8 weeks
            if run_args.future_tc is not None:
                window_size = 14
                change_over_n_windows = 6
                future_tcs = list(np.linspace(model.tc[-1], run_args.future_tc, change_over_n_windows + 1))[1:]
                future_tslices = list(range(model.tslices[-1], model.tslices[-1] + window_size*change_over_n_windows + 1, window_size))[1:]
                model.apply_tc(tc=future_tcs, tslices=future_tslices)

            # add winter TC shift starting on November 25
            if run_args.winter_tc_shift:
                today = dt.date.today()
                # shift TC at winter start date
                winter_start_date = dt.date(today.year, 11, 25) if today.strftime('%m%d') < '1125' else dt.date(today.year + 1, 11, 25)
                winter_start_t = (winter_start_date - model.start_date).days
                winter_tslices = [winter_start_t, *(tslice for tslice in model.tslices if tslice > winter_start_t)]
                winter_tc = [tc + run_args.winter_tc_shift for tc in model.tc[-len(winter_tslices):]]
                model.apply_tc(tc=winter_tc, tslices=winter_tslices)
                # shift TC back at winter end date
                winter_end_date = dt.date(today.year, 2, 1) if today.strftime('%m%d') < '0201' else dt.date(today.year + 1, 2, 1)
                winter_end_t = (winter_end_date - model.start_date).days
                post_winter_tslices = [winter_end_t, *(tslice for tslice in model.tslices if tslice > winter_end_t)]
                post_winter_tc = [tc - run_args.winter_tc_shift for tc in model.tc[-len(post_winter_tslices):]]
                model.apply_tc(tc=post_winter_tc, tslices=post_winter_tslices)
                print(post_winter_tslices)
                print(post_winter_tc)
                print(model.tslices)
                print(model.tc)

            # make immunity last forever for vaccinated people
            model.build_param_lookups()
            model.set_param('imm_decay_days', 999999999999, trange=range(919, model.tmax), except_attrs={'vacc': 'none'})

            # prep model
            model.prep(rebuild_param_lookups=False)

            t1 = perf_counter()
            print(f'Model prepped in {t1-t0} seconds.')

            # solve and plot
            model.solve_seir()
            label = (model.tags['scenario'] if 'scenario' in model.tags else 'unknown') + ', ' + sublabel
            modeled(model, 'Ih', ax=ax, label=label)

    format_date_axis(ax)
    plt.legend(loc='best')
    plt.show()
