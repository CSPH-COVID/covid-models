select
    measure_date
    , trended_hospitalized as currently_hospitalized
from emresource.hospitalized_with_corrections
order by 1