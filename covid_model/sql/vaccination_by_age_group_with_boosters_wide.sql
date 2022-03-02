select
    measure_date
    , age_group as age
	, dose1_mrna + dose1_jnj as shot1
	, dose2_mrna as shot2
	, booster as shot3
from vaccination.combined_doses_by_age_group v;