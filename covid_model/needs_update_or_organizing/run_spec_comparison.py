### Python Standard Library ###
from time import perf_counter
import argparse
import datetime as dt
### Third Party Imports ###
import numpy as np
import matplotlib.pyplot as plt
### Local Imports ###
from covid_model.analysis.charts import plot_modeled, plot_observed_hosps, format_date_axis
from covid_model.db import db_engine
from covid_model.model_with_future_variant import CovidModelWithFutureVariant


if __name__ == '__main__':
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

    fig, axs = plt.subplots(2)
    plot_observed_hosps(engine, ax=axs[1], color='black')

    for spec_id in run_args.spec_ids:

        t0 = perf_counter()
        print(f'Prepping model for specification {spec_id}...')
        model = CovidModelWithFutureVariant(end_date=to_date, engine=engine, from_specs=spec_id, params=run_args.params, future_seed_date=future_variant_seed_date)
        # model = CovidModel(end_date=to_date, engine=engine, from_specs=spec_id, params=run_args.params)

        # adjust TC to future TC over the next 8 weeks
        if run_args.future_tc is not None:
            window_size = 14
            change_over_n_windows = 6
            future_tcs = list(np.linspace(model.tc[-1], run_args.future_tc, change_over_n_windows + 1))[1:]
            future_tslices = list(range(model.tc_tslices[-1], model.tc_tslices[-1] + window_size * change_over_n_windows + 1, window_size))[1:]
            model.update_tc(tc=future_tcs, tslices=future_tslices)

        # add winter TC shift starting on November 25
        if run_args.winter_tc_shift:
            today = dt.date.today()
            # shift TC at winter start date
            winter_start_date = dt.date(today.year, 11, 25) if today.strftime('%m%d') < '1125' else dt.date(today.year + 1, 11, 25)
            winter_start_t = (winter_start_date - model.start_date).days
            winter_tslices = [winter_start_t, *(tslice for tslice in model.tc_tslices if tslice > winter_start_t)]
            winter_tc = [tc + run_args.winter_tc_shift for tc in model.tc[-len(winter_tslices):]]
            model.update_tc(tc=winter_tc, tslices=winter_tslices)
            # shift TC back at winter end date
            winter_end_date = dt.date(today.year, 2, 1) if today.strftime('%m%d') < '0201' else dt.date(today.year + 1, 2, 1)
            winter_end_t = (winter_end_date - model.start_date).days
            post_winter_tslices = [winter_end_t, *(tslice for tslice in model.tc_tslices if tslice > winter_end_t)]
            post_winter_tc = [tc - run_args.winter_tc_shift for tc in model.tc[-len(post_winter_tslices):]]
            model.update_tc(tc=post_winter_tc, tslices=post_winter_tslices)

        # prep model
        model.prep()

        t1 = perf_counter()
        print(f'Model prepped in {t1-t0} seconds.')

        # solve and plot
        model.solve_seir()
        label = model.tags['scenario'] if 'scenario' in model.tags else 'unknown'
        plot_modeled(model, ['I', 'A'], ax=axs[0], label=label, share_of_total=True)
        plot_modeled(model, 'Ih', ax=axs[1], label=label)

    for ax in axs:
        format_date_axis(ax, interval_months=1)
        ax.set_xlim(from_date, to_date)
        ax.legend(loc='best')

    axs[0].set_ylabel('SARS-CoV-2 Infection Prevalence')
    axs[1].set_ylabel('Hospitalized with COVID-19')
    plt.show()
