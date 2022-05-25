### Python Standard Library ###
import datetime as dt
from time import perf_counter
### Third Party Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
### Local Imports ###
from covid_model.analysis.charts import plot_modeled, plot_actual_hosps, format_date_axis
from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.cli_specs import ModelSpecsArgumentParser
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
    parser.add_argument("-fd", "--from_date", default='2021-07-01', type=str, help="min date for plots")
    run_args = parser.parse_args()
    from_date = dt.date.fromisoformat(run_args.from_date)
    to_date = run_args.end_date

    run_args.plot = ['prev', 'hosp', 'var', 'imm'] if run_args.plot is None else run_args.plot
    plots = [plot for plot in run_args.plot if plot in plot_opts.keys()]
    print("Will produce these plots:" + ", ".join([plot_opts[plot] for plot in plots]))

    print('Prepping model...')
    t0 = perf_counter()
    engine = db_engine()
    model = CovidModel(engine=engine, **parser.specs_args_as_dict())
    # model.apply_tc(tc=model.tc + [1.0], tslices=model.tslices + [815])
    model.prep()
    t1 = perf_counter()
    print(f'Model prepped in {t1-t0} seconds.')

    print('Running model...')
    model.solve_seir()
    # build_legacy_output_df(model).to_csv('output/out2.csv')

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
            plot_actual_hosps(engine, ax=ax, color='black')
            # data added below under tc scenarios section
        if plot == "var":
            # variants
            plot_modeled(model, ['I', 'A'], groupby='variant', share_of_total=True, ax=ax)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylabel('Variant Share of Infections')
            # ax.lines.pop(0)
            ax.legend(loc='best')
        if plot == "imm":
            # immunity
            ax.plot(model.daterange, model.immunity('omicron'), label='Immunity vs Omicron', color='cyan')
            ax.plot(model.daterange, model.immunity('omicron', age='65+'), label='Immunity vs Omicron (65+ only)', color='darkcyan')
            ax.plot(model.daterange, model.immunity('omicron', to_hosp=True), label='Immunity vs Severe Omicron', color='gold')
            ax.plot(model.daterange, model.immunity('omicron', to_hosp=True, age='65+'), label='Immunity vs Severe Omicron (65+ only)', color='darkorange')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylim(0, 1)
            ax.set_ylabel('Percent Immune')
            ax.legend(loc='best')
            ax.set_xlim(from_date, to_date)
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
            ax.set_xlim(from_date, to_date)

    if ("prev" in plots) or ("hosp" in plots):
        # tc shift scenarios
        base_tslices = model.tc_tslices.copy()
        base_tc = model.tc.copy()
        if "hosp" in plots:
            hosps_df = pd.DataFrame(index=model.t_eval)
        # for tc_shift, tc_shift_days in [(0, 0), (-0.05, 14), (-0.1, 14), (-0.2, 21), (-0.5, 42)]:
        for tc_shift, tc_shift_days in [(0, 0)]:
            # TODO: Make the start date for TC shifts dynamic and/or configurable
            start_t = 749
            future_tslices = list(range(start_t, start_t + tc_shift_days))
            future_tc = np.linspace(base_tc[-1], base_tc[-1] + tc_shift, len(future_tslices))
            model.apply_tc(tcs=base_tc + list(future_tc), tslices=base_tslices + list(future_tslices))
            model.solve_seir()
            label = f'{round(100*-tc_shift)}% drop in TC over {round(len(future_tslices)/7)} weeks' if tc_shift < 0 else f'Current trajectory'
            if "hosp" in plots:
                plot_modeled(model, 'Ih', ax=ax_hosp, label=label)
                hosps_df[label] = model.solution_sum('seir')['Ih']
            if "prev" in plots:
                plot_modeled(model, ['I', 'A'], share_of_total=True, ax=ax_prev, label=label)
        if "hosp" in plots:
            hosps_df.index = model.daterange
            hosps_df.loc[:'2022-02-28'].round(1).to_csv('output/omicron_report_hospitalization_scenarios.csv')

    # formatting
    for ax in axs:
        format_date_axis(ax)
        ax.set_xlim(from_date, to_date)
        ax.axvline(x=dt.date.today(), color='darkgray')
        ax.grid(color='lightgray')
        ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig('output/omicron_report.png')
    plt.show()
