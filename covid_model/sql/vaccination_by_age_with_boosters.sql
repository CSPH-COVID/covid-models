select
    measure_date
    , age_group as age
    , sum(dose1_mrna) + sum(dose1_jnj) as shot1
    , sum(dose2_mrna) as shot2
    , sum(booster1) as booster1
    , sum(booster2) as booster2
    , sum(booster3) as booster3
from `co-covid-models.vaccination.doses_by_county_by_age_group_withbooster3`
where county_id in UNNEST(%(county_ids)s)
group by measure_date, age
order by measure_date, age