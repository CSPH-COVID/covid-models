from collections import OrderedDict

from covid_model.model import CovidModel
from covid_model.model_specs import CovidModelSpecifications

import matplotlib.pyplot as plt
from covid_model.analysis.charts import modeled, actual_hosps, mmwr_infections_growth_rate, re_estimates
from time import perf_counter
from timeit import timeit

from covid_model.db import db_engine
from covid_model.model import CovidModel
from covid_model.model_specs import CovidModelSpecifications


class CovidModelWithVariants(CovidModel):

    attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                        'priorinf': ['none', 'non-omicron', 'omicron'],
                        'variant': ['none', 'omicron'],
                        'immun': ['none', 'imm1', 'imm2', 'imm3']})

    param_attr_names = ('age', 'vacc', 'priorinf', 'variant', 'immun')

    # do not apply vaccines in the traditional way; we'll cover them using attribute_multipliers instead
    def prep(self, specs=None, **specs_args):
        t0 = perf_counter()
        self.set_specifications(specs=specs, **specs_args)
        t1 = perf_counter()
        print(t1 - t0)
        t0 = perf_counter()
        self.apply_specifications(apply_vaccines=False)
        t1 = perf_counter()
        print(t1 - t0)
        t0 = perf_counter()
        self.apply_new_vacc_params()
        t1 = perf_counter()
        print(t1 - t0)
        t0 = perf_counter()
        self.build_ode()
        t1 = perf_counter()
        print(t1 - t0)
        t0 = perf_counter()
        self.compile()
        t1 = perf_counter()
        print(t1 - t0)
        t0 = perf_counter()

    def apply_new_vacc_params(self):
        vacc_per_available = self.specifications.get_vacc_per_available()
        vacc_fail_per_vacc = self.specifications.get_vacc_fail_per_vacc()

        # convert to dictionaries for performance lookup
        vacc_per_available_dict = vacc_per_available.to_dict()

        # set the fail rate and vacc per unvacc rate for each dose
        for shot in self.attr['vacc'][1:]:
            for age in self.attr['age']:
                self.set_param(f'{shot}_fail_rate', vacc_fail_per_vacc[shot][age], {'age': age})
                for t in self.trange:
                    self.set_param(f'{shot}_per_available', vacc_per_available_dict[shot][(t, age)], {'age': age}, trange=[t])

    # build ODE
    def build_ode(self):
        self.reset_ode()
        # build S to E and R to E elements of the ODE
        self.build_SR_to_E_ode()
        # add flows by attributes
        # vaccination
        for i in range(1, len(self.attributes['vacc'])):
            self.add_flows_by_attr({'vacc': f'shot{i-1}' if i >= 2 else 'none'}, {'vacc': f'shot{i}', 'immun': f'imm{i}'}, coef=f'shot{i}_per_available * (1 - shot{i}_fail_rate)')
            self.add_flows_by_attr({'vacc': f'shot{i-1}' if i >= 2 else 'none'}, {'vacc': f'shot{i}'}, coef=f'shot{i}_per_available * shot{i}_fail_rate')
        # disease progression
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'I'}, coef='1 / alpha * pS')
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'A'}, coef='1 / alpha * (1 - pS)')
        self.add_flows_by_attr({'seir': 'I'}, {'seir': 'Ih'}, coef='gamm * hosp * (1 - severe_immunity)')
        # disease termination
        for variant in self.attributes['variant']:
            priorinf = 'omicron' if variant == 'omicron' else 'non-omicron'
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'imm3'}, coef='gamm * (1 - hosp - dnh) * immune_rate_I')
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
            self.add_flows_by_attr({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'imm3'}, coef='gamm * immune_rate_A')
            self.add_flows_by_attr({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * (1 - immune_rate_A)')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'imm3'}, coef='1 / hlos * (1 - dh) * immune_rate_I')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * (1 - dh) * (1 - immune_rate_I)')
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * dnh * (1 - severe_immunity)')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * dh')
        # immunity decay
        self.add_flows_by_attr({'immun': 'imm3'}, {'immun': 'imm2'}, coef='1 / 90')
        self.add_flows_by_attr({'immun': 'imm2'}, {'immun': 'imm1'}, coef='1 / 360')
        self.add_flows_by_attr({'immun': 'imm1'}, {'immun': 'none'}, coef='1 / 540')

    def build_SR_to_E_ode(self):
        # seed omicron
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'omicron'}, constant='om_seed')
        # apply flow from S to E (note that S now encompasses recovered as well
        asymptomatic_transmission = '(1 - immunity) * betta * (1 - ef) / total_pop'
        for variant in self.attributes['variant']:
            sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant})
            asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant})
            # immunity = f'immunity * (1 - {f"{variant}_immunity_reduction" if variant in ("delta", "omicron") else "0"})'
            self.add_flows_by_attr({'seir': 'S'}, {'seir': 'E', 'variant': variant}, coef=f'lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
            self.add_flows_by_attr({'seir': 'S'}, {'seir': 'E', 'variant': variant}, coef=asymptomatic_transmission, scale_by_cmpts=asympt_cmpts)

    # define initial state y0
    @property
    def y0_dict(self):
        y0d = {('S', age, 'none', 'none', 'none', 'none'): n for age, n in self.specifications.group_pops.items()}
        y0d[('S', '40-64', 'none', 'none', 'none', 'none')] -= 2.2
        y0d[('I', '40-64', 'none', 'none', 'none', 'none')] = 2.2
        return y0d


if __name__ == '__main__':
    engine = db_engine()

    model = CovidModelWithVariants()

    print('Prepping model...')
    # print(timeit('model.prep(521, engine=engine)', number=1, globals=globals()), 'seconds to prep model.')
    model.prep(551, engine=engine, params='input/params.json', attribute_multipliers='input/attribute_multipliers.json')
    print('Running model...')
    print(timeit('model.solve_seir()', number=1, globals=globals()), 'seconds to run model.')

    fig, ax = plt.subplots()
    modeled(model, 'Ih')
    actual_hosps(engine)

    # print({k: v['shot1_per_available'] for k, v in model.params[500].items()})

    model.solution_sum('vacc').plot()
    model.solution_sum('immun').plot()
    plt.show()

