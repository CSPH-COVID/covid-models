### Python Standard Library ###
import random
from time import perf_counter
import datetime as dt
### Third Party Imports ###
import numpy as np
import pandas as pd
import scipy.stats as sps
import pmdarima
import arch
from sqlalchemy import MetaData
### Local Imports ###
from covid_model.db import db_engine
from covid_model import CovidModel


def forecast_timeseries(data, horizon=1, sims=10, arima_order='auto', use_garch=False):
    historical_values = np.log(1 - np.array(data))
    if arima_order is None or arima_order == 'auto':
       arima_model = pmdarima.auto_arima(historical_values, suppress_warnings=True, seasonal=False)
    else:
        arima_model = pmdarima.ARIMA(order=arima_order, suppress_warnings=True).fit(historical_values)
    arima_results = arima_model.arima_res_
    p, d, q = arima_model.order

    # fit ARIMA on transformed
    arima_residuals = arima_model.arima_res_.resid

    if use_garch:
        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch = arch.arch_model(arima_residuals, p=1, q=1)
        garch_model = garch.fit(disp='off')
        garch_sims = [e[0] for e in garch_model.forecast(horizon=1, reindex=False, method='simulation').simulations.values[0]]

        # simulate projections iteratively
        all_projections = []
        for i in range(sims):
            projections = []
            for steps_forward in range(horizon):
                projected_error = random.choice(garch_sims)
                projected_mean = arima_results.forecast(1)[0]

                projections.append(projected_mean + projected_error)
                arima_results = arima_results.append([projections[-1]])

            all_projections.append(projections)

    else:
        projections = arima_results.simulate(horizon, anchor='end', repetitions=sims)
        all_projections = [[projections[step][0][rep] for step in range(horizon)] for rep in range(sims)]

    return 1 - np.exp(np.array(all_projections))


class CovidModelSimulation:
    def __init__(self, specs, engine, end_date=None):
        self.model = CovidModel(end_date=end_date, from_specs=specs, engine=engine)
        self.model.prep()
        self.base_tc = self.model.tc.copy()

        self.window_size = self.model.tslices[-1] - self.model.tslices[-2]
        self.simulation_horizon = int(np.ceil((self.model.tmax - self.model.tslices[-1]) / self.window_size)) - 1

        tslices = self.model.tslices + list(range(self.model.tslices[-1] + self.window_size, self.model.tmax, self.window_size))
        tc = self.base_tc + [self.base_tc[-1]] * (len(tslices) - len(self.base_tc) + 1)
        self.model.apply_tc(tslices=tslices, tc=tc)

        self.engine = engine
        self.db_metadata = MetaData(schema='covid_model')
        self.db_metadata.reflect(engine, only=['simulations', 'simulation_results', 'results'])
        self.table = self.db_metadata.tables[f'covid_model.simulations']
        self.results_table = self.db_metadata.tables[f'covid_model.simulation_results']
        self.sim_id = None
        self.write_to_db(engine)

        self.base_results = None
        self.results = []
        self.results_hosps = []

    @classmethod
    def from_db(cls, engine, sim_id):
        df = pd.read_sql_query(f"select * from covid_model.simulations where sim_id = {sim_id}", con=engine, coerce_float=True)
        if len(df) == 0:
            raise ValueError(f'{sim_id} is not a valid sim ID.')
        row = df.iloc[0]
        sim = CovidModelSimulation(row['spec_id'], engine, end_date=row['end_date'])

        sim.results = pd.read_sql_query(f"select * from covid_model.simulation_results where sim_id = {sim_id}", con=engine, coerce_float=True)
        sim.results_hosps = sim.results.set_index(['sim_result_id', 't'])['Ih'].unstack('sim_result_id')
        sim.results_hosps.index = sim.model.daterange

        return sim

    def write_to_db(self, engine):

        stmt = self.table.insert().values(
            created_at=dt.datetime.now(),
            spec_id=int(self.model.spec_id),
            start_date=self.model.start_date,
            end_date=self.model.end_date,
        )

        conn = engine.connect()
        result = conn.execute(stmt)

        self.sim_id = result.inserted_primary_key[0]
        self.db_metadata.reflect(engine, only=['simulations', 'simulation_results', 'results'])

    def run_base_result(self):
        self.model.solve_seir()
        self.base_results = self.model.solution_ydf.stack(level=self.model.param_attr_names)
        self.model.write_results_to_db(self.engine)

    def sample_fitted_tcs(self, sample_n=1):
        fitted_count = len(self.model.tc_cov)
        fitted_efs_dist = sps.multivariate_normal(mean=self.base_tc[-fitted_count:], cov=[[float(x) for x in a] for a in self.model.tc_cov])
        fitted_efs_samples = fitted_efs_dist.rvs(sample_n)

        return [list(self.base_tc[:-fitted_count]) + list(sample if hasattr(sample, '__iter__') else [sample]) for sample in (fitted_efs_samples if sample_n > 1 else [fitted_efs_samples])]

    def sample_simulated_tcs(self, sample_n=1, sims_per_fitted_sample=5, arima_order='auto', skip_early_tcs=8):
        if len(np.unique(np.diff(self.model.tslices[skip_early_tcs-1:]))) > 1:
            raise ValueError('Window-sizes for TCs used for prediction must be evenly spaced.')

        simulated_tcs = []
        fitted_sample_n = int(np.ceil(sample_n / sims_per_fitted_sample))
        fitted_tcs_sample = self.sample_fitted_tcs(fitted_sample_n)

        for i, fitted_tcs in enumerate(fitted_tcs_sample):
            print(f'Generated {len(simulated_tcs)}/{sample_n} future TC values.')
            next_tcs_sample = forecast_timeseries(fitted_tcs[skip_early_tcs:], sims=sims_per_fitted_sample, horizon=self.simulation_horizon, arima_order=arima_order)
            simulated_tcs += [list(fitted_tcs) + list(next_tcs) for next_tcs in next_tcs_sample]

        return simulated_tcs[:sample_n]

    def run_simulations(self, n, **tc_sampling_args):
        print(f'Simulating future TC values...')
        simulated_tcs = self.sample_simulated_tcs(n, **tc_sampling_args)
        for i, tcs in enumerate(simulated_tcs):
            t0 = perf_counter()
            self.model.apply_tc(tcs)
            self.model.solve_seir()
            self.results.append(self.model.solution_ydf.stack(level=self.model.param_attr_names))
            self.results_hosps.append(self.model.solution_sum('seir')['Ih'])
            t1 = perf_counter()
            self.model.write_results_to_db(self.engine, sim_id=self.sim_id, sim_result_id=i, cmpts_json_attrs=tuple())
            t2 = perf_counter()
            print(f'Simulation {i+1}/{len(simulated_tcs)} completed in {round(t2-t0, 4)} sec, including {round(t2-t1, 4)} sec to write to database.')

        results_hosps_df = pd.DataFrame({i: hosps for i, hosps in enumerate(self.results_hosps)})
        hosp_percentiles = {int(100*qt): list(results_hosps_df.quantile(0.05, axis=1).values) for qt in [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]}

        stmt = self.table.update_specs().where(self.table.c.sim_id == self.sim_id).values(
            sim_count=len(self.results_hosps),
            hospitalized_percentiles=hosp_percentiles
        )

        conn = self.engine.connect()
        result = conn.execute(stmt)


if __name__ == '__main__':
    engine = db_engine()

    sims = CovidModelSimulation(303, engine=engine, end_date=dt.date(2022, 5, 31))
    sims.run_simulations(3)
