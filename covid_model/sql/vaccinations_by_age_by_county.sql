select
    v.measure_date::date as measure_date
    , c.county_id
    , c.county_label as county
    , case
        when v.age <= 19 then '0-19'
        when v.age >= 20 and v.age <= 39 then '20-39'
        when v.age >= 40 and v.age <= 64 then '40-64'
        when v.age >= 65 then '65+'
    end as age
    , coalesce(round(sum(v.first_dose_rate)), 0) as first_doses_given
    , coalesce(round(sum(v.first_dose_rate - v.jnj_dose_rate)), 0) as mrna_first_doses_given
    , coalesce(round(sum(v.final_dose_rate - v.jnj_dose_rate)), 0) as mrna_second_doses_given
    , coalesce(round(sum(v.jnj_dose_rate)), 0) as jnj_doses_given
    , coalesce(round(sum(v.booster_dose_rate)), 0) as booster_doses_given
from cdphe.covid19_vaccinations_by_age_by_county v
join geo_dev.us_counties c using (county_id)
group by 1, 2, 3, 4
order by 1, 2, 3, 4;