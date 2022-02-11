with combined as (
	select
		dd::date as measure_date
		, cph.countofpatientsinhospitalwithcovid as hospitalized_cophs
		, emr.currently_hospitalized as hospitalized_emr
	from generate_series('2020-01-24'::date, (select max(measure_date::date) from cdphe.emresource_hospitalizations), interval '1 day') dd
	left join stage.cophs cph on dd = cph.dates::date
	left join cdphe.emresource_hospitalizations emr on dd = emr.measure_date::date
)
select
	measure_date
	, case
		when cophs_max - measure_date > 30 then coalesce(hospitalized_cophs, 0)
		else round(hospitalized_emr * (select 1. * sum(hospitalized_cophs) / sum(hospitalized_emr) as trailing_cophs_emr_ratio from combined where cophs_max - measure_date between 31 and 60))
	end as currently_hospitalized
from combined
	, (select max(dates::date) as cophs_max from stage.cophs) cm


--select
--	dates::date as measure_date
--	, round(0.84 * countofpatientsinhospitalwithcovid) as currently_hospitalized
--from stage.cophs
--where dates::date < '2020-04-20'
--union
--select
--	measure_date
--	, currently_hospitalized
--from cdphe.emresource_hospitalizations eh
--where measure_date >= '2020-04-20'
--order by 1