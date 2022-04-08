### Python Standard Library ###
import datetime as dt
### Third Party Imports ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
### Local Imports ###
from covid_model.db import db_engine
from covid_model.data_imports import get_vaccinations


def build_data(engine, proj_params, group_pops, sql, group_pops_alt=None):
    df = get_vaccinations(engine
                          , from_date=dt.datetime(2020, 12, 1)
                          , proj_to_date=dt.datetime(2021, 12, 31)
                          , proj_lookback=7
                          , proj_fixed_rates=proj_params['fixed_rates'] if 'fixed_rates' in proj_params.keys() else None
                          , max_cumu={g: group_pops[g] * proj_params['max_cumu'][g] for g in group_pops.keys()}
                          , max_rate_per_remaining=proj_params['max_rate_per_remaining']
                          , realloc_priority=proj_params['realloc_priority']
                          , sql=sql)

    if group_pops_alt is None:
        group_pops_alt = group_pops
    df = df.groupby(['measure_date', 'group']).sum()
    df['cumu'] = df['rate'].groupby(['group']).cumsum()
    df = df.join(pd.Series(group_pops_alt).rename('population').rename_axis('group'))
    df['cumu_share'] = df['cumu'] / df['population']
    return df


if __name__ == '__main__':
    engine = db_engine()

    old_proj_params = {
      "lookback": 14,
      "max_cumu": {"0-19": 0.2134, "20-39": 0.50, "40-64": 0.62, "65+": 0.94},
      "max_rate_per_remaining": 0.04,
      "realloc_priority": ['65+', '40-64', '20-39', '0-19']}

    old_group_pops = {
        "0-19": 1513005,
        "20-39": 1685869,
        "40-64": 1902963,
        "65+": 738958}

    old_sql = """select
        reporting_date as measure_date
        , count_type as "group"
        , case date_type when 'vaccine dose 1/2' then 'mrna' when 'vaccine dose 1/1' then 'jnj' end as vacc
        , sum(total_count) as rate
    from cdphe.covid19_county_summary
    where date_type like 'vaccine dose 1/_' and reporting_date < '2021-05-12'
    group by 1, 2, 3
    order by 1, 2, 3"""

    proj_params = {
      "lookback": 7,
      "max_cumu": {"0-19": 0.215, "20-39": 0.62, "40-64": 0.72, "65+": 0.86},
      "max_rate_per_remaining": 0.04,
      "realloc_priority": None}

    group_pops = {
        "0-19": 1413638,
        "20-39": 1701344,
        "40-64": 1820564,
        "65+": 877662}

    sql = open('sql/vaccinations_by_age_group.sql', 'r').read()

    df = build_data(engine, proj_params, group_pops, sql, group_pops_alt=group_pops)
    df = df[df['is_projected'] == 0]

    old_df = build_data(engine, old_proj_params, old_group_pops, old_sql, group_pops_alt=group_pops)
    old_df = old_df.loc[:df.index.max()]

    # plots
    def plot(ycol):
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ['steelblue', 'green', 'orangered', 'blueviolet']

        for color, (name, group) in zip(colors, old_df.reset_index().set_index('measure_date').groupby('group')):
            group.plot(ax=ax, y=ycol, label=f"{name} (low-uptake proj. from May '21)", color=color, alpha=0.33)

        for color, (name, group) in zip(colors, df.reset_index().set_index('measure_date').groupby('group')):
            group.plot(ax=ax, y=ycol, label=f'{name} (actual)', color=color)

        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        # ax.ticklabel_format(useOffset=False)
        # ax.ticklabel_format(style='plain')
        ax.get_yaxis().set_major_formatter(
            mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_xlabel(None)
        ax.set_ylabel('Percent of Total Vaccinated Using New Denominators')
        ax.set_ylabel('People Vaccinated w/ At Least One Dose')
        # ax.set_title('Percent of Total Vaccinated, Projected vs. Actual (new pop. est.)')
        ax.set_title('People Vaccinated, Projected vs. Actual')
        plt.grid(True, alpha=0.5)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    print(df.index.max())
    # df['cumu'] = '{:,}'.format(df['cumu'])
    plot('cumu')
