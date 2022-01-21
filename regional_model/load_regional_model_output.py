import pandas as pd
import datetime as dt
import os
from db_utils.conn import db_engine


def run():
    dfs = {}
    dir = 'input'
    fnames = os.listdir('input')
    lpha_fname = max(fn for fn in fnames if fn[:14] == 'MetroMobOutput')
    metro_fname = max(fn for fn in fnames if fn[:13] == 'LPHAMobOutput')
    fit_date = dt.date(2021, int(lpha_fname[-9:-7]), int(lpha_fname[-7:-5]))
    for fname in [lpha_fname, metro_fname]:
        dfs_by_region = pd.read_excel(os.path.join(dir, fname), engine='openpyxl', sheet_name=None)
        df = pd.concat(dfs_by_region).reset_index(level=1, drop=True).set_index('Date', append=True)
        df['fit_date'] = dt.date(2021, int(fname[-9:-7]), int(fname[-7:-5]))
        dfs[fname.split('MobOutput')[0]] = df

    combined = pd.concat(dfs)
    combined.index.names = ['region_type', 'region', 'measure_date']
    column_mapping = {
        'Day': 't'
        , 'Hospitalizations': 'hosp'
        , 'HospPer100000': 'hosp_per_100k'
        , 'Incidence': 'incidence'
        , 'pImmune': 'immun_share'
        , 'PrevPer100000': 'prev_per_100k'
        , 'CumulativeInfToDate': 'cumu_inf'
        , 'ReEstimate': 're'
        , 'fit_date': 'fit_date'}
    combined = combined.rename(columns=column_mapping)[column_mapping.values()]
    combined['fit_date'] = fit_date

    engine = db_engine()
    combined.to_sql('regional_model_results', engine, schema='stage', method='multi', chunksize=1000, if_exists='replace')


if __name__ == '__main__':
    run()
