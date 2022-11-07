select county_id
     , measure_date
     , hospitalized as observed_hosp
from cophs.hospitalized_by_county
where county_id in unnest(%(county_ids)s)
order by 1, 2
