select Region
     , week as measure_date
     , InpatientBedsCovid7daysum_calc as observed
from cste.hospitalizations_by_region
where Region in unnest(%(region_ids)s)
order by 1, 2