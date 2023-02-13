SELECT
       measure_date,
       region_id AS region,
       age_group AS age,
       dose1 AS shot1,
       dose2 AS shot2,
       booster1,
       booster2,
       booster3
FROM `cste_testing.cdc_state_doses_by_region_by_age_group`
WHERE region_id = %(region_id)s
ORDER BY measure_date,region_id,age_group
