import numpy as np
import pandas as pd
import datetime as dt
from time import perf_counter

import scipy.optimize as spo

from data_imports import ExternalHosps
from model import CovidModel
from model_specs import CovidModelSpecifications


class CovidModelFit:

    def __init__(self, base_specs, engine=None, tc_0=0.75, tc_min=0, tc_max=0.99, new_end_date=dt.date(2022, 5, 31)):
        if isinstance(base_specs, CovidModelSpecifications):
            self.base_specs = base_specs
        elif isinstance(base_specs, int):
            self.base_specs = CovidModelSpecifications.from_db(engine, base_specs, new_end_date=new_end_date)
        else:
            raise TypeError(f'Invalid type for base_specs: {type(base_specs)}')

        self.tc_0 = tc_0
        self.tc_min = tc_min
        self.tc_max = tc_max

        self.actual_hosp = None

        self.fitted_tc = None
        self.fitted_tc_cov = None

    def set_actual_hosp_from_db(self, engine):
        self.actual_hosp = ExternalHosps(engine, t0_date=self.base_specs.start_date).fetch('emresource_hosps.csv')['currently_hospitalized']

    def single_fit(self, model: CovidModel, look_back, method='curve_fit'):
        fitted_tc, fitted_tc_cov = (None, None)
        fixed_tc = model.specifications.tc[:-look_back]
        if method == 'curve_fit':
            def func(trange, *test_tc):
                combined_tc = fixed_tc + list(test_tc)
                model.apply_tc(combined_tc)
                model.solve_seir()
                return model.solution_sum('seir')['Ih']
            fitted_tc, fitted_tc_cov = spo.curve_fit(
                f=func
                , xdata=model.trange
                , ydata=self.actual_hosp[:len(model.trange)]
                , p0=model.specifications.tc[-look_back:]
                , bounds=([self.tc_min] * look_back, [self.tc_max] * look_back))

        return fitted_tc, fitted_tc_cov

    # run an optimization to minimize the cost function using scipy.optimize.minimize()
    # method = 'curve_fit' or 'minimize'
    def run(self, engine, method='curve_fit', window_size=14, look_back=3, last_window_min_size=21, batch_size=None, increment_size=1):

        # get the end date from actual hosps
        end_t = self.actual_hosp.index.max() + 1
        end_date = self.base_specs.start_date + dt.timedelta(end_t)

        # prep model (we only do this once to save time)
        t0 = perf_counter()
        base_model = CovidModel(start_date=self.base_specs.start_date, end_date=end_date)
        base_model.prep(specs=self.base_specs)
        t1 = perf_counter()
        print(f'Model prepped for fitting in {t1-t0} seconds.')

        # run fit
        tslices = self.base_specs.tslices + list(range(self.base_specs.tslices[-1] + window_size, end_t - last_window_min_size, window_size))
        tc = self.base_specs.tc + [self.tc_0] * (len(tslices) + 1 - len(self.base_specs.tc))
        fitted_tc_cov = None
        if look_back is None:
            look_back = len(tslices) + 1

        # if there's no batch size, set the batch size to be the total number of windows to be fit
        if batch_size is None:
            batch_size = look_back

        trim_off_end_list = list(range(look_back - batch_size, 0, -increment_size)) + [0]
        for i, trim_off_end in enumerate(trim_off_end_list):
            t0 = perf_counter()
            this_end_t = tslices[-trim_off_end] if trim_off_end > 0 else end_t
            this_end_date = self.base_specs.start_date + dt.timedelta(days=this_end_t)
            model = CovidModel(start_date=self.base_specs.start_date, end_date=this_end_date)
            model.specifications = self.base_specs.copy(this_end_date)
            model.params = base_model.params
            model.apply_tc(tc[:len(tc)-trim_off_end], tslices=tslices[:len(tslices)-trim_off_end])
            model.build_ode()

            fitted_tc, fitted_tc_cov = self.single_fit(model, look_back=batch_size, method=method)
            tc[len(tc) - trim_off_end - batch_size:len(tc) - trim_off_end] = fitted_tc

            t1 = perf_counter()
            print(f'Transmission control fit {i + 1}/{len(trim_off_end_list)} completed in {t1 - t0} seconds.')

        self.fitted_tc = tc
        self.fitted_tc_cov = fitted_tc_cov
        self.fitted_model = model
        self.fitted_specs = model.specifications
        self.fitted_specs.set_tc(tc=tc, tslices=tslices, tc_cov=fitted_tc_cov)





