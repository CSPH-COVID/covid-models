from collections import OrderedDict

from covid_model.model import CovidModel


class CovidModelWithVariants(CovidModel):

    attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['unvacc', 'vacc', 'vacc_fail'],
                        'pinf': ['none', 'pinf_fail', 'pinf1', 'pinf2', 'pinf_decayed'],
                        'variant': ['none', 'omicron']})

    param_attr_names = ('age', 'vacc')

    # build ODE
    def build_ode(self):
        self.reset_ode()
        # build S to E and R to E elements of the ODE
        self.build_SR_to_E_ode()
        # build vacc and disease lifecycle
        for variant in self.attributes['variant']:
            for age in self.attributes['age']:
                for pinf in self.attributes['pinf']:
                    # vaccination
                    for seir in self.attributes['seir']:
                        self.add_flow((seir, age, 'unvacc', pinf, variant), (seir, age, 'vacc_fail', variant), 'shot1_per_unvacc * shot1_fail_rate')
                        self.add_flow((seir, age, 'unvacc', pinf, variant), (seir, age, 'vacc', variant), 'shot1_per_unvacc * (1 - shot1_fail_rate)')
                        self.add_flow((seir, age, 'vacc_fail', pinf, variant), (seir, age, 'vacc', variant), 'vacc_fail_reduction_per_vacc_fail')
                    for vacc in self.attributes['vacc']:
                        # disease
                        self.add_flow(('E', age, vacc, pinf, variant), ('I', age, vacc, pinf, variant), '1 / alpha * pS')
                        self.add_flow(('E', age, vacc, pinf, variant), ('A', age, vacc, pinf, variant), '1 / alpha * (1 - pS)')
                        self.add_flow(('I', age, vacc, pinf, variant), ('Ih', age, vacc, pinf, variant), 'gamm * hosp')
                        self.add_flow(('I', age, vacc, pinf, variant), ('D', age, vacc, pinf, variant), 'gamm * dnh')
                        self.add_flow(('I', age, vacc, pinf, variant), ('S', age, vacc, 'pinf1', 'none'), 'gamm * (1 - hosp - dnh) * immune_rate_I')
                        self.add_flow(('I', age, vacc, pinf, variant), ('S', age, vacc, 'pinf_fail', 'none'), 'gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
                        self.add_flow(('A', age, vacc, pinf, variant), ('S', age, vacc, 'pinf1', 'none'), 'gamm * immune_rate_A')
                        self.add_flow(('A', age, vacc, pinf, variant), ('S', age, vacc, 'pinf_fail', 'none'), 'gamm * (1 - immune_rate_A)')
                        self.add_flow(('Ih', age, vacc, pinf, variant), ('D', age, vacc, pinf, variant), '1 / hlos * dh')
                        self.add_flow(('Ih', age, vacc, pinf, variant), ('S', age, vacc, 'pinf1', 'none'), '1 / hlos * (1 - dh) * immune_rate_I')
                        self.add_flow(('Ih', age, vacc, pinf, variant), ('S', age, vacc, 'pinf_fail', 'none'), '1 / hlos * (1 - dh) * (1 - immune_rate_I)')
                        # recovery
                        self.add_flow(('S', age, vacc, 'pinf1', 'none'), ('S', age, vacc, 'pinf2'), '1 / immune_decay_days_1')
                        self.add_flow(('S', age, vacc, 'pinf2', 'none'), ('S', age, vacc, 'pinf_decayed'), '1 / immune_decay_days_2')

    def build_SR_to_E_ode(self):
        # seed omicron
        self.add_flow(('S', '40-64', 'unvacc', 'none', 'none'), ('E', '40-64', 'unvacc', 'none', 'omicron'), constant='om_seed')
        # build ode
        vacc_eff_w_delta = 'vacc_eff * (1 - (1 - nondelta_prevalence) * delta_max_efficacy_reduction * (1 - vacc_eff))'
        base_transm = f'betta * (1 - ef) * (1 - {vacc_eff_w_delta}) / total_pop'
        for variant in self.attributes['variant']:
            infectious_cmpts = [(s, a, v, pinf, variant) for a in self.attributes['age'] for v in self.attributes['vacc'] for s in ['I', 'A'] for pinf in self.attributes['pinf']]
            infectious_cmpt_coefs = [' * '.join([
                'lamb' if seir == 'I' else '1',
                'omicron_transm_mult' if variant == 'omicron' else '1',
                'unvacc_relative_transm' if vacc == 'unvacc' else '1'
            ]) for seir, age, vacc, pinf, variant in
                infectious_cmpts]
            for age in self.attributes['age']:
                # transmission to susceptibles
                for pinf in self.attributes['pinf']:
                    self.add_flow(('S', age, 'unvacc', pinf, 'none'), ('E', age, 'unvacc', pinf, variant), f'unvacc_relative_transm * {base_transm}', scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                    self.add_flow(('S', age, 'vacc', pinf, 'none'), ('E', age, 'vacc', pinf, variant), base_transm, scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                    self.add_flow(('S', age, 'vacc_fail', pinf, 'none'), ('E', age, 'vacc_fail', pinf, variant), base_transm, scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                # transmission to recovered (a.k.a. acquired-immune escape)
                # note that currently all recovered people go back to variant "none"; change this if we want to reduce cross-variant immunity
                if variant == 'omicron':
                    self.add_flow(('S', age, 'unvacc', pinf, 'none'), ('E', age, 'unvacc', pinf, variant), f'omicron_acq_immune_escape * unvacc_relative_transm * {base_transm}', scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
                    self.add_flow(('R', age, 'vacc', 'none'), ('E', age, 'vacc', variant),
                                  f'omicron_acq_immune_escape * {base_transm}', scale_by_cmpts=infectious_cmpts,
                                  scale_by_cmpts_coef=infectious_cmpt_coefs)
                    self.add_flow(('R', age, 'vacc_fail', 'none'), ('E', age, 'vacc_fail', variant),
                              f'omicron_acq_immune_escape * {base_transm}', scale_by_cmpts=infectious_cmpts,
                              scale_by_cmpts_coef=infectious_cmpt_coefs)

    # reset terms that depend on TC; this takes about 0.08 sec, while rebuilding the whole ODE takes ~0.90 sec
    def rebuild_ode_with_new_tc(self):
        self.reset_terms({'seir': 'S'}, {'seir': 'E'})
        self.reset_terms({'seir': 'R'}, {'seir': 'E'})
        self.build_SR_to_E_ode()

    # define initial state y0
    @property
    def y0_dict(self):
        y0d = {('S', age, 'unvacc', 'none'): n for age, n in self.specifications.group_pops.items()}
        y0d[('I', '40-64', 'unvacc', 'none')] = 2.2
        y0d[('S', '40-64', 'unvacc', 'none')] -= 2.2
        return y0d

    # don't try to write to db, since the table format doesn't match
    def write_to_db(self, engine=None, new_spec=False):
        pass