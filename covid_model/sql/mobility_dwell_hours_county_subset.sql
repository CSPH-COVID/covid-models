select
    measure_date
    , origin_county_id
    , destination_county_id
    , total_dwell_duration_hrs
from `co-covid-models.mobility.county_to_county_by_week`
where origin_county_id in unnest(%(county_ids)s) and destination_county_id in unnest(%(county_ids)s)
order by 1, 2, 3