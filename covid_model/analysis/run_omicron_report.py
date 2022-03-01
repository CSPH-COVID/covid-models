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
from covid_model.cli_specs import ModelSpecsArgumentParser
# from covid_model.model_with_omicron import CovidModelWithVariants
# from covid_model.model_with_immunity_rework import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications
from covid_model.run_model_scenarios import build_legacy_output_df

plot_opts = {
    "prev": "SARS-CoV-2 Prevalence",
    "hosp": "Hospitalized with COVID-19",
    "var": "Variant Share of Infections",
    "imm": "Percent Immune"
}

if __name__ == '__main__':
    parser = ModelSpecsArgumentParser()
    parser.add_argument("--plot", action="append", choices=plot_opts.keys(), required=False,
                        help="add a plot to the output figure, default: Prevalence, Hospitalizations, Variant Share, "
                             "and Percent Immune")
    run_args = parser.parse_args()

    run_args.plot = ['prev', 'hosp', 'var', 'imm'] if run_args.plot is None else run_args.plot
    plots = [plot for plot in run_args.plot if plot in plot_opts.keys()]
    print("Will produce these plots:" + ", ".join([plot_opts[plot] for plot in plots]))

    print('Prepping model...')
    engine = db_engine()
    model = CovidModel(end_date=dt.date(2022, 10, 31), engine=engine, **parser.specs_args_as_dict())
    model.prep()
    model.apply_tc(tc=model.tc + [1.0], tslices=model.tslices + [850])

    print('Running model...')
    model.solve_seir()
    build_legacy_output_df(model).to_csv('output/out2.csv')

    print('Producing charts...')
    ncols = int(np.ceil(np.sqrt(len(plots))))
    nrows = int(np.ceil(len(plots)/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols+1, 8*nrows))

    axs = axs.flatten() if len(plots) > 1 else [axs]

    ax_prev = None
    ax_hosp = None
    # looping allows for plotting in the order specified
    for i, plot in enumerate(plots):
        ax = axs[i]
        if plot == "prev":
            # prevalence
            ax_prev = ax
            ax.set_ylabel('SARS-CoV-2 Prevalenca')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.legend(loc='best')
            # data added below under tc scenarios section
        if plot == "hosp":
            # hospitalizations
            ax_hosp = ax
            ax.set_ylabel('Hospitalized with COVID-19')
            ax.legend(loc='best')
            actual_hosps(engine, ax=ax, color='black')
            # data added below under tc scenarios section
        if plot == "var":
            # variants
            modeled(model, ['I', 'A'], groupby='variant', share_of_total=True, ax=ax)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylabel('Variant Share of Infections')
            ax.lines.pop(0)
            ax.legend(loc='best')
        if plot == "imm":
            # immunity
            ax.plot(model.daterange, model.immunity('none'), label='Immunity vs non-Omicron', color='cyan')
            ax.plot(model.daterange, model.immunity('omicron'), label='Immunity vs Omicron', color='darkcyan')
            ax.plot(model.daterange, model.immunity('none', vacc_only=True), label='Immunity vs non-Omicron (Vaccine-only)', color='gold')
            ax.plot(model.daterange, model.immunity('omicron', vacc_only=True), label='Immunity vs Omicron (Vaccine-only)', color='darkorange')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylim(0, 1)
            ax.set_ylabel('Percent Immune')
            ax.legend(loc='best')
            ax.set_xlim(dt.date(2020, 4, 1), dt.date(2022, 3, 31))
        if plot == "sevimm":
            # immunity
            ax.plot(model.daterange, model.immunity('none'), label='Immunity vs Severe non-Omicron', color='cyan')
            ax.plot(model.daterange, model.immunity('omicron'), label='Immunity vs Severe Omicron', color='darkcyan')
            ax.plot(model.daterange, model.immunity('none', vacc_only=True), label='Immunity vs Severe non-Omicron (Vaccine-only)', color='gold')
            ax.plot(model.daterange, model.immunity('omicron', vacc_only=True), label='Immunity vs Severe Omicron (Vaccine-only)', color='darkorange')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylim(0, 1)
            ax.set_ylabel('Percent Immune')
            ax.legend(loc='best')
            ax.set_xlim(dt.date(2020, 4, 1), dt.date(2022, 3, 31))

    if ("prev" in plots) or ("hosp" in plots):
        # tc shift scenarios
        base_tslices = model.tslices.copy()
        base_tc = model.tc.copy()
        if "hosp" in plots:
            hosps_df = pd.DataFrame(index=model.trange)
        for tc_shift, tc_shift_days in [(0, 0), (-0.05, 14), (-0.1, 14), (-0.2, 21), (-0.5, 42)]:
            # TODO: Make the start date for TC shifts dynamic and/or configurable
            start_t = 749
            future_tslices = list(range(start_t, start_t + tc_shift_days))
            future_tc = np.linspace(base_tc[-1], base_tc[-1] + tc_shift, len(future_tslices))
            model.apply_tc(tc=base_tc + list(future_tc), tslices=base_tslices + list(future_tslices))
            model.solve_seir()
            label = f'{round(100*-tc_shift)}% drop in TC over {round(len(future_tslices)/7)} weeks' if tc_shift < 0 else f'Current trajectory'
            if "hosp" in plots:
                modeled(model, 'Ih', ax=ax_hosp, label=label)
                hosps_df[label] = model.solution_sum('seir')['Ih']
            if "prev" in plots:
                modeled(model, ['I', 'A'], share_of_total=True, ax=ax_prev, label=label)
        if "hosp" in plots:
            hosps_df.index = model.daterange
            hosps_df.loc[:'2022-02-28'].round(1).to_csv('output/omicron_report_hospitalization_scenarios.csv')

    # formatting
    for ax in axs:
        format_date_axis(ax)
        ax.set_xlim(dt.date(2021, 7, 1), dt.date(2022, 2, 28))
        ax.axvline(x=dt.date.today(), color='darkgray')
        ax.grid(color='lightgray')
        ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig('output/omicron_report.png')
    plt.show()
