select
    measure_date::date as measure_date
    , case
	    when age <= 19 then '0-19'
	    when age >= 20 and age <= 39 then '20-39'
	    when age >= 40 and age <= 64 then '40-64'
	    when age >= 65 then '65+'
	end as "age"
	, sum(first_dose_rate) as shot1
	, sum(final_dose_rate) as shot2
	, sum(booster_dose_rate) as shot3
from cdphe.covid19_vaccinations_by_age_by_county v
where county_id in %(county_ids)s
group by 1, 2;
