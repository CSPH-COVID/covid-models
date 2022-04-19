### Python Standard Library ###
import datetime as dt
import json
### Third Party Imports ###
import pandas as pd
import numpy as np
### Local Imports ###
from covid_model.db import db_engine
from covid_model.data_imports import get_vaccinations_by_county, ExternalVaccWithProjections, ExternalHosps


if __name__ == '__main__':
    engine = db_engine()


    # export actual hospitalizations
    print('Exporting hospitalizations...')
    hosps = ExternalHosps(engine).fetch(county_ids=None)['currently_hospitalized'].reset_index().rename(columns={'currently_hospitalized': 'Iht', 'measure_date': 'date'})
    hosps['time'] = ((pd.to_datetime(hosps['date']) - dt.datetime(2020, 1, 24)) / np.timedelta64(1, 'D')).astype(int)
    hosps[['time', 'date', 'Iht']].to_csv('output/CO_EMR_Hosp.csv', index=False)

    # export vaccinations, with projections
    gparams = json.load(open('input/params.json', 'r'))
    proj_param_dict = json.load(open('input/vacc_proj_params.json', 'r'))
    vacc_df_dict = {}
    for label, proj_params in proj_param_dict.items():
        print(f'Exporting vaccination by age for "{label}" scenario...')
        df = ExternalVaccWithProjections(engine, fill_from_date=dt.datetime(2020, 1, 24),
                                         fill_to_date=dt.datetime(2022, 2, 28)).fetch('input/past_and_projected_vaccinations.csv',
                                                                        proj_params=proj_params, group_pop=gparams['group_pop'])
        # df['is_projected'] = df['is_projected'].fillna(False).astype(int)
        vacc_df_dict[label] = df.groupby(['measure_date', 'age']).sum().rename(columns={'rate': 'first_shot_rate'})
        # vacc_df_dict[label]['is_projected'] = vacc_df_dict[label]['is_projected'] > 0

    vacc_df = pd.concat(vacc_df_dict).rename_axis(index=['vacc_scen', 'measure_date', 'age']).sort_index()
    vacc_df['first_shot_cumu'] = vacc_df['shot1'].groupby(['vacc_scen', 'age']).cumsum()
    shots = vacc_df.columns
    print(shots)
    for shot in shots:
        vacc_df[f'{shot}_cumu'] = vacc_df[shot].groupby(['vacc_scen', 'age']).cumsum()
    vacc_df = vacc_df.join(pd.Series(gparams['group_pop']).rename('population').rename_axis('age'))
    vacc_df['cumu_share_of_population'] = vacc_df['first_shot_cumu'] / vacc_df['population']
    # vacc_df = vacc_df.join(vacc_df[~vacc_df['is_projected']]['first_shot_cumu'].groupby(['vacc_scen', 'age']).max().rename('current_first_shot_cumu'))
    vacc_df.to_csv('output/daily_vaccination_by_age.csv', float_format='%.15f')

    # export vaccination by county, without projections
    print('Exporting vaccination by county...')
    vacc_by_county = get_vaccinations_by_county(engine)
    vacc_by_county.to_csv('output/daily_vaccination_by_age_by_county.csv', index=False)

