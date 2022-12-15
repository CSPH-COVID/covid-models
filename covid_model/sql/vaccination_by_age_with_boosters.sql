select
    measure_date
    , age_group as age
    , sum(dose1_mrna) + sum(dose1_jnj) as shot1
    , sum(dose2_mrna) as shot2
    , sum(booster1) as booster1
    , sum(booster2) as booster2
from `co-covid-models.cste_testing.doses_by_county_by_cste_age_group`
where county_id in UNNEST(%(county_ids)s)
group by measure_date, age
order by measure_date, age