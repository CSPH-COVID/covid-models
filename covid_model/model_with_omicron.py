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
        # build vacc and disease lifecycle
        for variant in self.attributes['variant']:
            for age in self.attributes['age']:
                # vaccination
                for seir in self.attributes['seir']:
                    self.add_flow((seir, age, 'unvacc', variant), (seir, age, 'vacc_fail', variant), 'shot1_per_unvacc * shot1_fail_rate')
                    self.add_flow((seir, age, 'unvacc', variant), (seir, age, 'vacc', variant), 'shot1_per_unvacc * (1 - shot1_fail_rate)')
                    self.add_flow((seir, age, 'vacc_fail', variant), (seir, age, 'vacc', variant), 'vacc_fail_reduction_per_vacc_fail')
                # disease and recovery lifecycle
                for vacc in self.attributes['vacc']:
                    self.add_flow(('E', age, vacc, variant), ('I', age, vacc, variant), '1 / alpha * pS')
                    self.add_flow(('E', age, vacc, variant), ('A', age, vacc, variant), '1 / alpha * (1 - pS)')
                    self.add_flow(('I', age, vacc, variant), ('Ih', age, vacc, variant), 'gamm * hosp')
                    self.add_flow(('I', age, vacc, variant), ('D', age, vacc, variant), 'gamm * dnh')
                    self.add_flow(('I', age, vacc, variant), ('R', age, vacc, variant), 'gamm * (1 - hosp - dnh) * immune_rate_I')
                    self.add_flow(('I', age, vacc, variant), ('S', age, vacc, 'none'), 'gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
                    self.add_flow(('A', age, vacc, variant), ('R', age, vacc, variant), 'gamm * immune_rate_A')
                    self.add_flow(('A', age, vacc, variant), ('S', age, vacc, 'none'), 'gamm * (1 - immune_rate_A)')
                    self.add_flow(('Ih', age, vacc, variant), ('D', age, vacc, variant), '1 / hlos * dh')
                    self.add_flow(('Ih', age, vacc, variant), ('R', age, vacc, variant), '1 / hlos * (1 - dh) * immune_rate_I')
                    self.add_flow(('Ih', age, vacc, variant), ('S', age, vacc, 'none'), '1 / hlos * (1 - dh) * (1 - immune_rate_I)')
                    self.add_flow(('R', age, vacc, variant), ('R2', age, vacc, variant), '1 / immune_decay_days_1')
                    self.add_flow(('R2', age, vacc, variant), ('S', age, vacc, variant), '1 / immune_decay_days_2')

    def build_SR_to_E_ode(self):
        # seed omicron
        self.add_flow(('S', '40-64', 'unvacc', 'none'), ('E', '40-64', 'unvacc', 'omicron'), constant='om_seed')
        # build ode
        vacc_eff_w_delta = '(vacc_eff * nondelta_prevalence + vacc_eff_vs_delta * (1 - nondelta_prevalence))'
        base_transm = f'betta * (1 - ef) * (1 - {vacc_eff_w_delta}) / total_pop'
        base_transm_omicron = f'betta * (1 - ef) * (1 - omicron_vacc_eff) / total_pop'
        for variant in self.attributes['variant']:
            infectious_cmpts = [(s, a, v, variant) for a in self.attributes['age'] for v in self.attributes['vacc'] for
                                s in ['I', 'A']]
            infectious_cmpt_coefs = [' * '.join(['lamb' if seir == 'I' else '1']) for seir, age, vacc, variant in infectious_cmpts]
            for age in self.attributes['age']:
                # transmission to susceptibles
                self.add_flow(('S', age, 'unvacc', 'none'), ('E', age, 'unvacc', variant), base_transm_omicron if variant == 'omicron' else base_transm,
                              scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                self.add_flow(('S', age, 'vacc', 'none'), ('E', age, 'vacc', variant), base_transm_omicron if variant == 'omicron' else base_transm,
                              scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                self.add_flow(('S', age, 'vacc_fail', 'none'), ('E', age, 'vacc_fail', variant), base_transm_omicron if variant == 'omicron' else base_transm,
                              scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                # transmission to recovered (a.k.a. acquired-immune escape)
                if variant == 'omicron':
                    for recovered_cmpt in ('R', 'R2'):
                        self.add_flow((recovered_cmpt, age, 'unvacc', 'none'), ('E', age, 'unvacc', variant),
                                      f'omicron_acq_immune_escape * {base_transm_omicron}',
                                      scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                        self.add_flow((recovered_cmpt, age, 'vacc', 'none'), ('E', age, 'vacc', variant),
                                      f'omicron_acq_immune_escape * {base_transm_omicron}', scale_by_cmpts=infectious_cmpts,
                                      scale_by_cmpts_coef=infectious_cmpt_coefs)
                        self.add_flow((recovered_cmpt, age, 'vacc_fail', 'none' ), ('E', age, 'vacc_fail', variant),
                                      f'omicron_acq_immune_escape * {base_transm_omicron}', scale_by_cmpts=infectious_cmpts,
                                      scale_by_cmpts_coef=infectious_cmpt_coefs)

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