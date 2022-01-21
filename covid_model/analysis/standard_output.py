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

from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.model_with_omicron import CovidModelWithVariants
from covid_model.model_specs import CovidModelSpecifications

if __name__ == '__main__':
    engine = db_engine()

    model = CovidModelWithVariants()

    print('Prepping model...')
    model.prep(536, engine=engine, params='input/params.json', attribute_multipliers='input/attribute_multipliers.json')

    print('Running model...')
    model.solve_seir()

    print('Producing charts...')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs = axs.flatten()

    # prevalence
    modeled(model, ['I', 'A'], share_of_total=True, ax=axs[0])
    axs[0].set_ylabel('SARS-CoV-2 Prevalence')
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # hosps
    modeled(model, 'Ih', ax=axs[1])
    actual_hosps(engine, ax=axs[1], color='black')
    axs[1].set_ylabel('Hospitalized with COVID-19')

    # variants
    modeled(model, ['I', 'A'], groupby='variant', share_of_total=True, ax=axs[2])
    axs[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[2].set_ylabel('Variant Share of Infections')
    axs[2].lines.pop(0)

    # tc shift scenarios
    # for tc_shift in [0, -0.05, -0.1, -0.15, -0.2]:
    #     future_tc = model.specifications.tc[-1] + tc_shift
    #     model.apply_tc(tc=model.specifications.tc + [future_tc], tslices=model.specifications.tslices + [738])
    #     model.solve_seir()
    #     modeled(model, 'Ih', ax=axs[3], label=f'{round(100*-tc_shift)}% drop in TC on February 1' if tc_shift < 0 else f'Hold TC constant at {round(100*-tc_shift)}%')
    #
    # actual_hosps(engine, ax=axs[3], color='black')
    # axs[3].set_ylabel('Hospitalized with COVID-19')
    # axs[3].legend(loc='best')

    # immunity
    axs[3].plot(model.daterange, model.immunity('none'), label='Immunity vs non-Omicron')
    axs[3].plot(model.daterange, model.immunity('omicron'), label='Immunity vs Omicron')
    axs[3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[3].set_ylim(0, 1)
    axs[3].set_ylabel('Percent Immune')
    axs[3].legend(loc='best')

    # formatting
    for ax in axs:
        format_date_axis(ax)
        ax.set_xlim(dt.date(2021, 7, 1), dt.date(2022, 3, 31))
        ax.grid(color='lightgray')

    axs[3].set_xlim(dt.date(2020, 4, 1), dt.date(2022, 3, 31))

    fig.tight_layout()
    plt.show()
