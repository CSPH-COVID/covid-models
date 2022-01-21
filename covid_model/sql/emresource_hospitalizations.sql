select
	dates::date as measure_date
	, round(0.84 * countofpatientsinhospitalwithcovid) as currently_hospitalized
from stage.cophs
where dates::date < '2020-04-20'
union
select
	measure_date
	, currently_hospitalized
from cdphe.emresource_hospitalizations eh
where measure_date >= '2020-04-20'
order by 1