### Python Standard Library ###
import datetime as dt
from time import perf_counter
### Third Party Imports ###
from matplotlib import pyplot as plt
import scipy.optimize as spo
### Local Imports ###
from covid_model.data_imports import ExternalHosps
from covid_model import CovidModel
from covid_model.model_specs import CovidModelSpecifications
from covid_model.analysis.charts import transmission_control, modeled


class CovidModelFit:

    def __init__(self, tc_0=0.75, tc_min=0, tc_max=0.99, **specs_build_args):
        self.base_specs = CovidModelSpecifications(**specs_build_args)

        self.tc_0 = tc_0
        self.tc_min = tc_min
        self.tc_max = tc_max

        self.actual_hosp = None

        self.fitted_tc = None
        self.fitted_tc_cov = None

    def set_actual_hosp(self, engine=None, county_ids=None):
        self.actual_hosp = ExternalHosps(engine, t0_date=self.base_specs.start_date).fetch(county_ids=county_ids)['currently_hospitalized']

    def single_fit(self, model: CovidModel, look_back, method='curve_fit', y0d=None):
        # define initial states
        fitted_tc, fitted_tc_cov = (None, None)
        fixed_tc = model.tc[:-look_back]
        if method == 'curve_fit':
            def func(trange, *test_tc):
                combined_tc = fixed_tc + list(test_tc)
                model.apply_tc(combined_tc)
                model.solve_seir(y0_dict=y0d)
                return model.solution_sum('seir')['Ih']
            fitted_tc, fitted_tc_cov = spo.curve_fit(
                f=func
                , xdata=model.t_eval
                , ydata=self.actual_hosp[:len(model.t_eval)]
                , p0=model.tc[-look_back:]
                , bounds=([self.tc_min] * look_back, [self.tc_max] * look_back))

        return fitted_tc, fitted_tc_cov

    # run an optimization to minimize the cost function using scipy.optimize.minimize()
    # method = 'curve_fit' or 'minimize'
    def run(self, engine, method='curve_fit', window_size=14, look_back=None,
            last_window_min_size=21, batch_size=None, increment_size=1, write_batch_output=False, model_class=CovidModel,
            model_args=dict(), forward_sim_each_batch=False, use_base_specs_end_date=False, print_prefix="", **unused_args):
        # get the end date from actual hosps
        end_t = (self.base_specs.end_date - self.base_specs.start_date).days if use_base_specs_end_date else self.actual_hosp.index.max() + 1
        end_date = self.base_specs.start_date + dt.timedelta(end_t)

        # create base model
        base_model = model_class(from_specs=self.base_specs, end_date=end_date, **model_args)
        base_model.tags = {}

        # prep model (we only do this once to save time)
        t0 = perf_counter()
        tslices = self.base_specs.tslices + list(range(self.base_specs.tslices[-1] + window_size, end_t - last_window_min_size, window_size))
        tc = self.base_specs.tc + [self.tc_0] * (len(tslices) + 1 - len(self.base_specs.tc))
        base_model.apply_tc(tc=tc, tslices=tslices)
        base_model.prep()
        t1 = perf_counter()
        print(f'{print_prefix} Model prepped for fitting in {t1-t0} seconds.')

        # run fit
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

            t01 = perf_counter()
            model = model_class(base_model=base_model, end_date=this_end_date, deepcopy_params=False, **model_args)
            model.apply_tc(tc[:len(tc)-trim_off_end], tslices=tslices[:len(tslices)-trim_off_end])
            t02 = perf_counter()
            print(f'{print_prefix} Model copied in {t02-t01} seconds.')

            fitted_tc, fitted_tc_cov = self.single_fit(model, look_back=batch_size, method=method)
            tc[len(tc) - trim_off_end - batch_size:len(tc) - trim_off_end] = fitted_tc

            t1 = perf_counter()
            print(f'{print_prefix} Transmission control fit {i + 1}/{len(trim_off_end_list)} completed in {t1 - t0} seconds.')
            if write_batch_output:
                model.tags['run_type'] = 'intermediate-fit'
                model.write_to_db(engine)

            # simulate the model and save a picture of the output
            if forward_sim_each_batch:
                # solved tc's get applied during model fitting
                model.solve_seir()
                fig = plt.figure(figsize=(10, 10), dpi=300)
                ax = fig.add_subplot(211)
                hosps_df = self.actual_hosp[:len(model.daterange)]
                hosps_df.index = model.daterange
                hosps_df.plot(**{'color': 'red', 'label': 'Actual Hosps.'})
                modeled(model, compartments='Ih', c='blue', ax=ax)
                ax = fig.add_subplot(212)
                transmission_control(model, ax=ax)
                plt.savefig(f'covid_model/output/{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_forward_sim_{i}.png')
                plt.close()

        self.fitted_tc = tc
        self.fitted_tc_cov = fitted_tc_cov
        self.fitted_model = model
