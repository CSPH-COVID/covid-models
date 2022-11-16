select
    measure_date
    , age_group as age
	, dose1_mrna + dose1_jnj as shot1
	, dose2_mrna as shot2
	, booster1 as booster1
	, booster2 as booster2
	, booster3 as booster3
from vaccination.doses_by_age_group_with_booster3 v;