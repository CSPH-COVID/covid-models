SELECT region_id,
       Region as name,
       County as counties,
       LPAD(CAST(FIPS AS STRING),5,"0") as counties_fips
FROM cste.regions