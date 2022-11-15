select
    measure_date,
    case
        when age <= 19 then '0-19'
        when age <= 39 then '20-39'
        when age <= 64 then '40-64'
        when age >= 65 then '65+'
    end as age
    , sum(dose1_mrna) + sum(dose1_jnj) as shot1
    , sum(dose2_mrna) as shot2
    , sum(booster1) as booster1
    , sum(booster2) as booster2
from `co-covid-models.vaccination.combined_doses_by_age_group_with_booster3`
where county_id in unnest(%(county_ids)s)
group by measure_date, age
order by measure_date, age