select
    measure_date
    , trended_hospitalized as currently_hospitalized
from emresource.hospitalized_with_corrections
where measure_date <= (SELECT MAX(measure_date) FROM emresource.hospitalized)
order by 1