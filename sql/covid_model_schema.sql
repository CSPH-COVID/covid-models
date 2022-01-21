-- stage.covid_model_fits definition
CREATE TABLE stage.covid_model_fits (
	id serial NOT NULL,
	tslices _int4 NULL,
	model_params jsonb NULL,
	fit_params jsonb NULL,
	efs _float8 NULL,
	created_at timestamp(0) NULL,
	fit_label varchar NULL,
	tags jsonb NULL,
	observed_efs _float8 NULL,
	efs_cov _float8 NULL,
	CONSTRAINT covid_model_fits_pkey PRIMARY KEY (id)
);

-- stage.covid_model_results definition
CREATE TABLE stage.covid_model_results (
	created_at timestamp NULL,
	fit_id int4 NULL,
	"group" varchar NULL,
	t int4 NULL,
	ef float8 NULL,
	"S" float8 NULL,
	"E" float8 NULL,
	"I" float8 NULL,
	"II" float8 NULL,
	"Ih" float8 NULL,
	"Ic" float8 NULL,
	"A" float8 NULL,
	"R" float8 NULL,
	"RA" float8 NULL,
	"Rh" float8 NULL,
	"Rc" float8 NULL,
	"V" float8 NULL,
	"D" float8 NULL,
	observed_ef float8 NULL,
	vacc_prev_s float8 NULL,
	vacc_prev_e float8 NULL,
	vacc_prev_i float8 NULL,
	vacc_prev_h float8 NULL,
	vacc_prev_d float8 NULL
);
