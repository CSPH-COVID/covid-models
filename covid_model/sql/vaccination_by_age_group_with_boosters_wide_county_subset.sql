select
    measure_date
    , age_group as age
	, sum(dose1_mrna + dose1_jnj) as shot1
	, sum(dose2_mrna) as shot2
	, sum(booster1) as booster1
    , sum(booster2) as booster2
    , sum(booster3) as booster3
from vaccination.combined_doses_by_county_by_age_group
where county_id in unnest(%(county_ids)s)
group by 1, 2;