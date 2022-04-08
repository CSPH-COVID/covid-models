### Python Standard Library ###
### Third Party Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
### Local Imports ###
from covid_model.db import db_engine


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return mcolors.LinearSegmentedColormap('colormap',cdict,1024)


cases_sql = """
with cases_by_region as (
    select
        cc.reporting_date as measure_date
        , r.region
        , max(r.population) as population
        , sum(total_count) filter (where count_type = 'cases' and date_type = 'reported') as cases
        , 100000. * sum(total_count) filter (where count_type = 'cases' and date_type = 'reported') / max(r.population) as cases_per_100k
        , sum(total_count) filter (where count_type = 'positive PCR tests')
        	/ nullif(sum(total_count) filter (where count_type = 'PCR tests'), 0) as positivity
    from cdphe.covid19_county_summary cc
    join stage.new_co_regions r on lpad(cc.county_fips_code::text, 5, '0') = any(r.county_ids)
    group by 1, 2
)
select
    *
    , avg(cases) over (partition by region order by measure_date rows between 6 preceding and 0 preceding) as cases_7d_avg
    , avg(cases) over (partition by region order by measure_date rows between 13 preceding and 0 preceding) as cases_14d_avg
    , avg(cases_per_100k) over (partition by region order by measure_date rows between 6 preceding and 0 preceding) as cases_per_100k_7d_avg
    , avg(cases_per_100k) over (partition by region order by measure_date rows between 13 preceding and 0 preceding) as cases_per_100k_14d_avg
    , avg(positivity) over (partition by region order by measure_date rows between 6 preceding and 0 preceding) as positivity_7d_avg
    , avg(positivity) over (partition by region order by measure_date rows between 13 preceding and 0 preceding) as positivity_14d_avg
from cases_by_region;
"""


wave_start_dates_dict = {
    'Central Mountains': '2021-12-14',
    'Metro': '2021-12-17',
    'Metro South': '2021-12-19',
    'Central': '2021-12-21',
    'Northeast': '2021-12-21',
    'West Central Partnership': '2021-12-22',
    'Northwest': '2021-12-27',
    'Southeast': '2022-01-02',
    'Southwest': '2021-12-27',
    'East Central': '2021-12-20',
    'San Luis Valley': '2021-12-28',
    'South Central': '2021-12-24',
    'Southeast Central': '2021-12-31'
}


if __name__ == '__main__':
    engine = db_engine()
    df = pd.read_sql(cases_sql, engine, parse_dates=['measure_date'], index_col=['measure_date', 'region'])

    cophs = pd.read_csv('cophs_hosps_by_lpha.csv', parse_dates=['measure_date'], index_col=['measure_date']).stack()
    df['hospitalized'] = cophs[:'2022-01-17'].rename_axis(index=['measure_date', 'region']).rename('hospitalized')
    df['hospitalized_per_100k'] = 100000 * df['hospitalized'] / df['population']

    wave_start_dates = pd.to_datetime(pd.Series(wave_start_dates_dict, name='wave_start_date'))
    wave_start_dates.index.name = 'region'
    days_since_wave_start = (df.index.get_level_values('measure_date') - pd.merge(df, wave_start_dates, left_on='region', right_index=True)['wave_start_date']).dt.days

    index = pd.MultiIndex.from_arrays([days_since_wave_start.values, days_since_wave_start.index.get_level_values('region')], names=('Days Since Beginning of Omicron Wave', 'Region'))
    df = pd.DataFrame(index=index, data=df.values, columns=df.columns)

    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    plt.legend(fontsize=80, title_fontsize='40')

    num_colors = len(df.index.unique('Region'))
    cm = plt.get_cmap('nipy_spectral')
    cm = cmap_map(lambda x: 0.75 * x + 0.2, cm)
    cNorm = mcolors.Normalize(vmin=0, vmax=num_colors - 1)
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=cm)
    for ax in axs.flatten():
        ax.grid(color='lightgray')
        ax.set_xticks(np.arange(0, 70, 7))
        ax.set_xlim(0, df.index.unique('Days Since Beginning of Omicron Wave').max())
        ax.legend(bbox_to_anchor=(1.04, 1))
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

    ax = axs.flatten()[0]
    ax.set_ylabel('New Cases per 100,000 (7-day average)')
    df['cases_per_100k_7d_avg'].unstack('Region').plot(ax=ax)

    ax = axs.flatten()[1]
    ax.set_ylabel('Test Positivity')
    df['positivity_7d_avg'].unstack('Region').plot(ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 0.4)

    ax = axs.flatten()[2]
    ax.set_ylabel('Hospitalized per 100k')
    # regions_subset = [r for r in wave_start_dates_dict.keys() if r not in ('West Central Partnership', 'San Luis Valley', 'South Central')]
    # print()
    df['hospitalized_per_100k'].unstack('Region').plot(ax=ax)
    remove_lines = [i for i, population in enumerate(df['population'].unique()) if population < 150000]
    for i in sorted(remove_lines, reverse=True):
        ax.lines.pop(i)
    ax.set_ylim(0, 70)

    ax = axs.flatten()[3]
    ax.set_ylabel('Hospitalized (relative to Day 0)')
    df['hospitalized_index'] = df['hospitalized'] / df['hospitalized'].xs(0, level = 'Days Since Beginning of Omicron Wave')
    df['hospitalized_index'].unstack('Region').plot(ax=ax)
    remove_lines = [i for i, population in enumerate(df['population'].unique()) if population < 150000]
    for i in sorted(remove_lines, reverse=True):
        ax.lines.pop(i)

    for ax in axs.flatten():
        ax.grid(color='lightgray')
        ax.set_xticks(np.arange(0, 70, 7))
        ax.set_xlim(0, df.index.unique('Days Since Beginning of Omicron Wave').max())
        ax.legend(bbox_to_anchor=(1.04, 1))
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

    axs.flatten()[2].set_xlim(0, 35)

    fig.tight_layout()
    plt.show()