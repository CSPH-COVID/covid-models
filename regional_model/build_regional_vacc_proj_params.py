import pandas as pd
import numpy as np
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
from db_utils.conn import db_engine
from sklearn.linear_model import LinearRegression


def main():
    conn = db_engine()

    sql = """select
        v.measure_date 
        , r.region 
        , case
            when v.age <= 19 then '0-19'
            when v.age >= 20 and v.age <= 39 then '20-39'
            when v.age >= 40 and v.age <= 64 then '40-64'
            when v.age >= 65 then '65+'
        end as group 
        , round(sum(v.first_dose_rate)) as first_dose_rate
        , sum(v.population) as population 
    from cdphe.covid19_vaccinations_by_age_by_county v
    join stage.co_lpha_regions r on v.county_id = any(r.county_ids)
    group by 1, 2, 3"""

    df = pd.read_sql(sql, conn, parse_dates=['measure_date']).set_index(['measure_date', 'region', 'group'])
    df['share_first_dose_cumu'] = df.groupby(['region', 'group'])['first_dose_rate'].cumsum() / df['population']
    gb = df.loc['2021-06-01':].groupby(['region', 'group'])
    for group in gb.groups.keys():
        df = gb.get_group(group)
        X = df[['share_first_dose_cumu']]
        y = df['first_dose_rate']
        model = LinearRegression()
        results = model.fit(X, y)
        print(results.coef_)


if __name__ == '__main__':
    main()