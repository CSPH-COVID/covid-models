select
    measure_date
    , hospitalized as currently_hospitalized
from emresource.hospitalized_complete
order by 1