# Colorado COVID Models

This repo contains a set of tools for running and processing the State of Colorado COVID-19 models, designed by the Colorado COVID-19 Modeling Group. Other versions of these models can be found in [another repo](https://github.com/agb85/covid-19). Model design documentation and weekly modeling team reports can be found [here](https://agb85.github.io/covid-19/).

## Statewide COVID-19 Model

#### Configuration and Data Dependencies

- [params.json](covid_model/params.json) - json containing model parameters
- [proportionvariantovertime.csv](covid_model/proportionvariantovertime.csv) - csv containing daily breakdown of variant prevalence; file path set in [params.json](covid_model/params.json)
- PostgreSQL database with the following tables:
  - cdphe.emresource_hospitalizations - source for real hospitalization data used for fitting and plotting
  - cdphe.covid19_county_summary - source for real vaccination data, used to set vaccination rate in model
  - stage.covid_model_fits - destination table containing the fitted parameters for past model runs
  - stage.covid_model_results - destination table containing the results of past model runs

#### Run Scripts
- Run fit to determine transmission control parameters using [run_fit.py](covid_model/run_fit.py) (`run_fit.py --help` for details regarding command line arguments)
- Populate data for relevant scenarios using [run_model_scenarios.py](covid_model/run_model_scenarios.py); scenarios include:
  - scenarios for vaccine uptake
  - scenarios for upcoming shifts in transmission control (magnitude and date of shift)
  
## Regional COVID-19 Model

Code for the regional COVID-19 Model can be found in [another repo](https://github.com/agb85/covid-19). This repo contains some tools to do minimal processing of the outputs of the Regional COVID-19 Model.
