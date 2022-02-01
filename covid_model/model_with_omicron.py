from collections import OrderedDict

from covid_model.model import CovidModel
from covid_model.model_specs import CovidModelSpecifications


class CovidModelWithVariants(CovidModel):

    attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'R', 'R2', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['unvacc', 'vacc', 'vacc_fail'],
                        'variant': ['none', 'omicron']})

    param_attr_names = ('age', 'vacc', 'variant')

    # add set a different vacc efficacy for Omicron
    def apply_omicron_vacc_eff(self):
        self.set_param('omicron_vacc_eff', 0, {'vacc': 'unvacc'})
        self.set_param('omicron_vacc_eff', 0, {'vacc': 'vacc_fail'})
        vacc_mean_efficacy_vs_omicron_dict = self.specifications.get_vacc_mean_efficacy(k='omicron_vacc_eff_k').to_dict()
        for age in self.attr['age']:
            for t in self.trange:
                self.set_param('omicron_vacc_eff', vacc_mean_efficacy_vs_omicron_dict[(t, age)], {'age': age, 'vacc': 'vacc'}, trange=[t])

    def apply_specifications(self, specs: CovidModelSpecifications = None):
        super().apply_specifications(specs)
        self.apply_omicron_vacc_eff()

    # build ODE
    def build_ode(self):
        self.reset_ode()
        # build S to E and R to E elements of the ODE
        self.build_SR_to_E_ode()
        # add flows by attributes
        # vaccination
        self.add_flows_by_attr({'vacc': 'unvacc'}, {'vacc': 'vacc_fail'}, coef='shot1_per_unvacc * shot1_fail_rate')
        self.add_flows_by_attr({'vacc': 'unvacc'}, {'vacc': 'vacc'}, coef='shot1_per_unvacc * (1 - shot1_fail_rate)')
        self.add_flows_by_attr({'vacc': 'vacc_fail'}, {'vacc': 'vacc'}, coef='vacc_fail_reduction_per_vacc_fail')
        # disease lifecycle
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'I'}, coef='1 / alpha * pS')
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'A'}, coef='1 / alpha * (1 - pS)')
        self.add_flows_by_attr({'seir': 'I'}, {'seir': 'Ih'}, coef='gamm * hosp')
        self.add_flows_by_attr({'seir': 'I'}, {'seir': 'D'}, coef='gamm * dnh')
        self.add_flows_by_attr({'seir': 'I'}, {'seir': 'R'}, coef='gamm * (1 - hosp - dnh) * immune_rate_I')
        self.add_flows_by_attr({'seir': 'I'}, {'seir': 'S', 'variant': 'none'}, coef='gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
        self.add_flows_by_attr({'seir': 'A'}, {'seir': 'R'}, coef='gamm * immune_rate_A')
        self.add_flows_by_attr({'seir': 'A'}, {'seir': 'S', 'variant': 'none'}, coef='gamm * (1 - immune_rate_A)')
        self.add_flows_by_attr({'seir': 'Ih'}, {'seir': 'D'}, coef='1 / hlos * dh')
        self.add_flows_by_attr({'seir': 'Ih'}, {'seir': 'R'}, coef='1 / hlos * (1 - dh) * immune_rate_I')
        self.add_flows_by_attr({'seir': 'Ih'}, {'seir': 'S', 'variant': 'none'}, coef='1 / hlos * (1 - dh) * (1 - immune_rate_I)')
        self.add_flows_by_attr({'seir': 'R'}, {'seir': 'R2'}, coef='1 / immune_decay_days_1')
        self.add_flows_by_attr({'seir': 'R2'}, {'seir': 'S'}, coef='1 / immune_decay_days_2')

    def build_SR_to_E_ode(self):
        # seed omicron
        self.add_flow(('S', '40-64', 'unvacc', 'none'), ('E', '40-64', 'unvacc', 'omicron'), constant='om_seed')
        # build ode
        vacc_eff_w_delta = '(vacc_eff * nondelta_prevalence + vacc_eff_vs_delta * (1 - nondelta_prevalence))'
        base_transm = f'betta * (1 - ef) * (1 - {vacc_eff_w_delta}) / total_pop'
        base_transm_omicron = f'betta * (1 - ef) * (1 - omicron_vacc_eff) / total_pop'
        for variant in self.attributes['variant']:
            sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant})
            asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant})
            self.add_flows_by_attr({'seir': 'S', 'variant': 'none'}, {'seir': 'E', 'variant': variant}, coef='lamb * ' + (base_transm_omicron if variant == 'omicron' else base_transm), scale_by_cmpts=sympt_cmpts)
            self.add_flows_by_attr({'seir': 'S', 'variant': 'none'}, {'seir': 'E', 'variant': variant}, coef=base_transm_omicron if variant == 'omicron' else base_transm, scale_by_cmpts=asympt_cmpts)
            if variant == 'omicron':
                for recovered_cmpt in ('R', 'R2'):
                    self.add_flows_by_attr({'seir': recovered_cmpt, 'variant': 'none'}, {'seir': 'E', 'variant': variant}, coef='omicron_acq_immune_escape * lamb * ' + (base_transm_omicron if variant == 'omicron' else base_transm), scale_by_cmpts=sympt_cmpts)
                    self.add_flows_by_attr({'seir': recovered_cmpt, 'variant': 'none'}, {'seir': 'E', 'variant': variant}, coef='omicron_acq_immune_escape * ' + (base_transm_omicron if variant == 'omicron' else base_transm), scale_by_cmpts=asympt_cmpts)

    # reset terms that depend on TC; this takes about 0.08 sec, while rebuilding the whole ODE takes ~0.90 sec
    def rebuild_ode_with_new_tc(self):
        self.reset_terms({'seir': 'S'}, {'seir': 'E'})
        self.reset_terms({'seir': 'R'}, {'seir': 'E'})
        self.reset_terms({'seir': 'R2'}, {'seir': 'E'})
        self.build_SR_to_E_ode()

    # define initial state y0
    @property
    def y0_dict(self):
        y0d = {('S', age, 'unvacc', 'none'): n for age, n in self.specifications.group_pops.items()}
        y0d[('I', '40-64', 'unvacc', 'none')] = 2.2
        y0d[('S', '40-64', 'unvacc', 'none')] -= 2.2
        return y0d