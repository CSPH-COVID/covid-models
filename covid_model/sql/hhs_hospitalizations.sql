select
    measure_date
    , confirmed_covid_hospitalizations as currently_hospitalized
from co-covid-models.cdc.co_hosp_hhs_statewidemodel v;
order by 1