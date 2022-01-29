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
from covid_model.model_with_omicron import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications
from covid_model.run_model_scenarios import build_legacy_output_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_id", type=int, required=True, help="the ID for the desired specifications")
    run_args = parser.parse_args()

    print('Prepping model...')
    engine = db_engine()
    model = CovidModelWithVariants()
    model.prep(run_args.spec_id, engine=engine, params='input/params.json', attribute_multipliers='input/attribute_multipliers.json')

    print('Running model...')
    model.solve_seir()
    build_legacy_output_df(model).to_csv('output/out2.csv')

    print('Producing charts...')
    fig, axs = plt.subplots(2, 2, figsize=(17, 8))

    axs = axs.flatten()

    # prevalence
    axs[0].set_ylabel('SARS-CoV-2 Prevalence')
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[0].legend(loc='best')

    # hospitalizations
    axs[1].set_ylabel('Hospitalized with COVID-19')
    axs[1].legend(loc='best')

    # variants
    modeled(model, ['I', 'A'], groupby='variant', share_of_total=True, ax=axs[2])
    axs[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[2].set_ylabel('Variant Share of Infections')
    axs[2].lines.pop(0)
    axs[2].legend(loc='best')

    # immunity
    axs[3].plot(model.daterange, model.immunity('none'), label='Immunity vs non-Omicron', color='cyan')
    axs[3].plot(model.daterange, model.immunity('omicron'), label='Immunity vs Omicron', color='darkcyan')
    axs[3].plot(model.daterange, model.immunity('none', vacc_only=True), label='Immunity vs non-Omicron (Vaccine-only)', color='gold')
    axs[3].plot(model.daterange, model.immunity('omicron', vacc_only=True), label='Immunity vs Omicron (Vaccine-only)', color='darkorange')
    axs[3].plot(model.daterange, model.immunity('omicron', to_hosp=True), label='Immunity vs Omicron Hospitalization', color='black')
    axs[3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[3].set_ylim(0, 1)
    axs[3].set_ylabel('Percent Immune')
    axs[3].legend(loc='best')

    actual_hosps(engine, ax=axs[1], color='black')

    # tc shift scenarios
    base_tslices = model.specifications.tslices.copy()
    base_tc = model.specifications.tc.copy()
    hosps_df = pd.DataFrame(index=model.trange)
    for tc_shift, tc_shift_days in [(0, 0), (-0.05, 14), (-0.1, 14), (-0.2, 21), (-0.5, 42)]:
        start_t = 725
        future_tslices = list(range(start_t, start_t + tc_shift_days))
        future_tc = np.linspace(base_tc[-1], base_tc[-1] + tc_shift, len(future_tslices))
        model.apply_tc(tc=base_tc + list(future_tc), tslices=base_tslices + list(future_tslices))
        model.solve_seir()
        label = f'{round(100*-tc_shift)}% drop in TC over {round(len(future_tslices)/7)} weeks' if tc_shift < 0 else f'Current trajectory'
        modeled(model, 'Ih', ax=axs[1], label=label)
        modeled(model, ['I', 'A'], share_of_total=True, ax=axs[0], label=label)
        hosps_df[label] = model.solution_sum('seir')['Ih']

    hosps_df.index = model.daterange
    hosps_df.loc[:'2022-02-28'].round(1).to_csv('output/omicron_report_hospitalization_scenarios.csv')

    # formatting
    for ax in axs:
        format_date_axis(ax)
        ax.set_xlim(dt.date(2021, 7, 1), dt.date(2022, 2, 28))
        ax.axvline(x=dt.date.today(), color='darkgray')
        ax.grid(color='lightgray')

    axs[3].set_xlim(dt.date(2020, 4, 1), dt.date(2022, 3, 31))

    fig.tight_layout()
    fig.savefig('output/omicron_report.png')
    plt.show()