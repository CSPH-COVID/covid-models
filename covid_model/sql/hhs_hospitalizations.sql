select
    measure_date
    , hospitalized as currently_hospitalized
from cdc.hhs_hospitalizations_complete
order by 1