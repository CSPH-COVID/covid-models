select region_id as Region
     , week as measure_date
     , total_covid as observed
from cste.hosps_by_region_with_synthetic
where region_id in unnest(%(region_ids)s)
order by Region, measure_date