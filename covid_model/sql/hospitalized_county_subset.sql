select
    measure_date
    , sum(hospitalized) as currently_hospitalized
from cophs.hospitalized_by_county
where county_id in unnest(%(county_ids)s)
group by 1
order by 1