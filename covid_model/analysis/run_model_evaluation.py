from covid_model.cli_specs import ModelSpecsArgumentParser
from covid_model.model import CovidModel
from covid_model.db import db_engine
from covid_model.data_imports import ExternalHosps
from covid_model.analysis.charts import modeled, actual_hosps
from covid_model.model_sims import forecast_timeseries

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    parser = ModelSpecsArgumentParser()
    parser.add_argument('-pd', '--projection_days', type=int, help='number of days to project into the future for retrospective evaluation')
    parser.add_argument('-skip', '--skip_first_increments', type=int, default=0, help='skip this many prediction increments at the beginning of the pandemic')
    run_args = parser.parse_args()

    fig, axs = plt.subplots(3)

    print('Prepping base model...')
    engine = db_engine()
    base_model = CovidModel(engine=engine, **parser.specs_args_as_dict())
    actual_hospitalizations = ExternalHosps(engine, t0_date=base_model.start_date).fetch(county_ids=None)['currently_hospitalized'].rename('Actual Hospitalizations')

    base_model.prep()
    base_model.solve_seir()

    print('Running retroactive projections...')
    projections = dict()
    for last_tc, projection_start_t in zip(base_model.tc[run_args.skip_first_increments:], base_model.tslices[run_args.skip_first_increments:]):
        # shorten model to end tslice + projection_days
        projection_end_t = min(projection_start_t + run_args.projection_days, base_model.tmax)
        projection_start_date = base_model.start_date + dt.timedelta(days=projection_start_t)
        projection_end_date = base_model.start_date + dt.timedelta(days=projection_end_t)

        model = CovidModel(base_model=base_model, end_date=projection_end_date)

        # set tc after tslice to the tc value immediately before tslice
        projected_tc_count = len([ts for ts in model.tslices if ts >= projection_start_t])
        fixed_tc = model.tc[:-projected_tc_count]
        projected_tc_dict = dict()
        projected_tc_dict['Hold TC Constant'] = [last_tc] * projected_tc_count
        projected_tc_dict['Revert TC to 75%'] = [0.75] * projected_tc_count
        projected_tc_dict['Timeseries Forecast'] = [x.mean() for x in np.array(forecast_timeseries(fixed_tc[run_args.skip_first_increments-2:], horizon=projected_tc_count, sims=1000, arima_order=(1, 0, 1))).transpose()]

        colors = ['navy', 'crimson', 'deepskyblue']
        projections_by_method = dict()
        for i, (tc_projection_method, projected_tc) in enumerate(projected_tc_dict.items()):
            print(f'{tc_projection_method}: projected TC from {projection_start_date} to {projection_end_date} is {projected_tc})')
            model.apply_tc(tc=projected_tc)

            # run model
            model.solve_seir()

            # write results to dictionary to dictionary
            projections_by_method[tc_projection_method] = model.solution_sum('seir')['Ih'].loc[projection_start_t:projection_end_t].rename('Projected Hospitalizations').to_frame()

        projections[projection_start_date] = pd.concat(projections_by_method, names=['Projection Method', 't'])

    # build dataframe
    projected_hospitalizations = pd.concat(projections, names=['Projection Start Date', 'Projection Method', 't'])
    df = pd.merge(projected_hospitalizations, actual_hospitalizations, left_on='t', right_index=True).reset_index()
    df['Date'] = base_model.start_date
    df['Date'] += pd.to_timedelta(df['t'], unit='d')
    df['Day of Projection'] = (df['Date'] - df['Projection Start Date']).dt.days
    df['Error'] = df['Projected Hospitalizations'] - df['Actual Hospitalizations']
    df['Sq. Error'] = np.square(df['Error'])

    # plot
    colors = ['navy', 'crimson', 'deepskyblue']
    sns.lineplot(data=df, x='Date', y='Projected Hospitalizations', hue='Projection Method', style='Projection Start Date', dashes=['']*len(df['Projection Start Date'].unique()), ax=axs[0], legend=False, palette=colors)
    sns.lineplot(data=np.sqrt(df.groupby(['Projection Method', 'Projection Start Date'])['Sq. Error'].mean()).to_frame(), x='Projection Start Date', y='Sq. Error', hue='Projection Method', ax=axs[1], palette=colors)
    sns.lineplot(data=np.sqrt(df.groupby(['Projection Method', 'Day of Projection'])['Sq. Error'].mean()).to_frame(), x='Day of Projection', y='Sq. Error', hue='Projection Method', ax=axs[2], palette=colors)
    # sns.heatmap(data=)
    actual_hosps(engine, color='black', ax=axs[0])

    axs[0].set_ylim(0, 4000)

    axs[0].set_ylabel('Hospitalized with COVID-19')
    axs[1].set_ylabel('Root-Mean-Square Error')
    axs[2].set_ylabel('Root-Mean-Square Error')

    for ax in axs.flatten():
        ax.grid('lightgray')
    plt.show()







