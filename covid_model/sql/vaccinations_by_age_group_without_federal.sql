select
    reporting_date as measure_date
    , count_type as "age"
    , case date_type when 'vaccine dose 1/2' then 'mrna' when 'vaccine dose 1/1' then 'jnj' end as vacc
    , sum(total_count) as rate
from cdphe.covid19_county_summary
where date_type like 'vaccine dose 1/_'
group by 1, 2, 3
order by 1, 2, 3