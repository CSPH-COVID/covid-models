select
    measure_date::date as measure_date
    , case
	    when age <= 19 then '0-19'
	    when age >= 20 and age <= 39 then '20-39'
	    when age >= 40 and age <= 64 then '40-64'
	    when age >= 65 then '65+'
	end as "age"
	, vacc
    , coalesce(round(case
    	when vacc = 'mrna' then sum(v.first_dose_rate - v.jnj_dose_rate)
    	when vacc = 'jnj' then sum(v.jnj_dose_rate)
    end), 0) as rate
from cdphe.covid19_vaccinations_by_age_by_county v
	, unnest(array['mrna', 'jnj']) vacc
group by 1, 2, 3
order by 1, 2, 3;