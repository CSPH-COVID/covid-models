select region_id as Region
     , week as measure_date
     , inpatient_beds_used_covid_7_day_sum as observed
from cste.hosps_by_region
where region_id in unnest(%(region_ids)s)
order by 1, 2