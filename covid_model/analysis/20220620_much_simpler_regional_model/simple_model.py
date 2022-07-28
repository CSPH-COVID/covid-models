""" Python Standard Library """
import logging
from collections import OrderedDict
""" Third Party Imports """
import numpy as np
import pandas as pd
""" Local Imports """
from covid_model.model import CovidModel
from covid_model.utils import get_params, IndentLogger, get_filepath_prefix, db_engine
logger = IndentLogger(logging.getLogger(''), {})


class SimpleCovidModel(CovidModel):
    def __init__(self, regions=['co'], **margs):
        super().__init__(**margs)
        self.attrs = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                                  'age': ['0-19', '20-39', '40-64', '65+'],
                                  'vacc': ['none', 'vaccinated'],
                                  'variant': ['none', 'preomicron', 'omicron'],
                                  'immun': ['none', 'immune'],
                                  'region': regions})
        self.update_data(engine=db_engine())

    def build_param_lookups(self, apply_vaccines=True, vacc_delay=14):
        logger.debug(f"{str(self.tags)} Building param lookups")
        # clear model params if they exist
        self.params_by_t = {'all': {}}

        for param_def in self.params_defs:
            if 'from_attrs' in param_def:
                self.set_from_to_compartment_param(**param_def)
            else:
                self.set_compartment_param(**param_def)

        # determine all times when params change
        self.params_trange = sorted(list(set.union(*[set(param.keys()) for param_key in self.params_by_t.values() for param in param_key.values()])))
        self.t_prev_lookup = {t_int: max(t for t in self.params_trange if t <= t_int) for t_int in self.trange}

        if apply_vaccines:
            logger.debug(f"{str(self.tags)} Building vaccination param lookups")
            vacc_per_available = self.get_vacc_per_available()

            # apply vacc_delay
            vacc_per_available = vacc_per_available.groupby(['region', 'age']).shift(vacc_delay).fillna(0)

            # group vacc_per_available by trange interval
            #bins = self.params_trange + [self.tend] if self.tend not in self.params_trange else self.params_trange
            bins = list(range(self.tstart, self.tend, 7))
            bins = bins + [self.tend] if self.tend not in bins else bins
            t_index_rounded_down_to_tslices = pd.cut(vacc_per_available.index.get_level_values('t'), bins, right=False, retbins=False, labels=bins[:-1])
            vacc_per_available = vacc_per_available.groupby([t_index_rounded_down_to_tslices, 'region', 'age']).mean()
            vacc_per_available['date'] = [self.t_to_date(d) for d in vacc_per_available.index.get_level_values(0)]
            vacc_per_available = vacc_per_available.reset_index().set_index(['region', 'age']).sort_index()

            # set the fail rate and vacc per unvacc rate for each dose
            for shot in ['shot1', 'shot2', 'shot3']:
                for age in self.attrs['age']:
                    for region in self.attrs['region']:
                        vpa_sub = vacc_per_available[['date', shot]].loc[region, age]
                        # TODO: hack for when there's only one date. Is there a better way?
                        if isinstance(vpa_sub, pd.Series):
                            vpa_sub = {vpa_sub[0]: vpa_sub[1]}
                        else:
                            vpa_sub = vpa_sub.reset_index(drop=True).set_index('date').sort_index().drop_duplicates().to_dict()[shot]
                        self.set_compartment_param(param=f'{shot}_per_available', attrs={'age': age, 'region': region}, vals=vpa_sub)

        # Testing changing the vaccination to every week
        self.params_trange = sorted(list(set.union(*[set(param.keys()) for param_key in self.params_by_t.values() for param in param_key.values()])))
        self.t_prev_lookup = {t_int: max(t for t in self.params_trange if t <= t_int) for t_int in self.trange}

    def build_ode_flows(self):
        logger.debug(f"{str(self.tags)} Building ode flows")
        self.flows_string = self.flows_string = '(' + ','.join(self.attr_names) + ')'
        self.reset_ode()

        # vaccination
        logger.debug(f"{str(self.tags)} Building vaccination flows")
        for seir in ['S', 'E', 'A']:
            for i in [2, 3]:
                for immun in self.attrs['immun']:
                    # TODO: this pretends like people getting shot 2 and shot 3 are different, which is not true. Really need to deal with both at the same time I think. Or ignore shot3
                    self.add_flows_from_attrs_to_attrs({'seir': seir, 'vacc': f'none', "immun": immun}, {'vacc': f'vaccinated', 'immun': f'immune'}, from_coef=f'shot{i}_per_available * (1 - shot{i}_fail_rate / shot{i - 1}_fail_rate)')
                    self.add_flows_from_attrs_to_attrs({'seir': seir, 'vacc': f'none', "immun": immun}, {'vacc': f'vaccinated', 'immun': f'none'}, from_coef=f'shot{i}_per_available * (shot{i}_fail_rate / shot{i - 1}_fail_rate)')

        # seed variants (only seed the ones in our attrs)
        logger.debug(f"{str(self.tags)} Building seed flows")
        for variant in self.attrs['variant']:
            if variant == 'none':
                continue
            seed_param = f'{variant}_seed'
            from_variant = self.attrs['variant'][0]   # first variant
            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': from_variant, 'immun': 'none'}, {'seir': 'E', 'variant': variant}, constant=seed_param)

        # exposure
        logger.debug(f"{str(self.tags)} Building transmission flows")
        for variant in self.attrs['variant']:
            if variant == 'none':
                continue
            # No mobility between regions (or a single region)
            if self.mobility_mode is None or self.mobility_mode == "none":
                for region in self.attrs['region']:
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef='lamb * betta', from_coef=f'(1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': region})
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef=       'betta', from_coef=f'(1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': region})
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef="lamb * betta", from_coef=f'immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': region})
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': region}, {'seir': 'E', 'variant': variant}, to_coef=       "betta", from_coef=f'immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': region})
            # Transmission parameters attached to the susceptible population
            elif self.mobility_mode == "population_attached":
                for infecting_region in self.attrs['region']:
                    for susceptible_region in self.attrs['region']:
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef='lamb * betta', from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       'betta', from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef="lamb * betta", from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       "betta", from_coef=f'mob_{susceptible_region}_exposure_from_{infecting_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})
            # Transmission parameters attached to the transmission location
            elif self.mobility_mode == "location_attached":
                for infecting_region in self.attrs['region']:
                    for susceptible_region in self.attrs['region']:
                        for transmission_region in self.attrs['region']:
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef='lamb * betta', from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       'betta', from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * (1 - immunity) * kappa / region_pop', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef="lamb * betta", from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'I', 'variant': variant, 'region': infecting_region})
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'region': susceptible_region}, {'seir': 'E', 'variant': variant}, to_coef=       "betta", from_coef=f'mob_{transmission_region}_frac_from_{infecting_region} * mob_{susceptible_region}_frac_in_{transmission_region} * immunity * kappa / region_pop', from_to_coef='immune_escape', scale_by_attrs={'seir': 'A', 'variant': variant, 'region': infecting_region})

        # disease progression
        logger.debug(f"{str(self.tags)} Building disease progression flows")
        self.add_flows_from_attrs_to_attrs({'seir': 'E'}, {'seir': 'I'}, to_coef='1 / alpha * pS')
        self.add_flows_from_attrs_to_attrs({'seir': 'E'}, {'seir': 'A'}, to_coef='1 / alpha * (1 - pS)')
        # assume no one is receiving both pax and mab
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, to_coef='gamm * hosp * (1 - severe_immunity) * (1 - mab_prev - pax_prev)')
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, to_coef='gamm * hosp * (1 - severe_immunity) * mab_prev * mab_hosp_adj')
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, to_coef='gamm * hosp * (1 - severe_immunity) * pax_prev * pax_hosp_adj')

        # disease termination
        logger.debug(f"{str(self.tags)} Building termination flows")
        for variant in self.attrs['variant']:
            if variant == 'none':
                continue
            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'S', 'immun': 'immune'}, to_coef='gamm * (1 - hosp - dnh) * (1 - priorinf_fail_rate)')
            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'S'}, to_coef='gamm * (1 - hosp - dnh) * priorinf_fail_rate')
            self.add_flows_from_attrs_to_attrs({'seir': 'A', 'variant': variant}, {'seir': 'S', 'immun': 'immune'}, to_coef='gamm * (1 - priorinf_fail_rate)')
            self.add_flows_from_attrs_to_attrs({'seir': 'A', 'variant': variant}, {'seir': 'S'}, to_coef='gamm * priorinf_fail_rate')

            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'immun': 'immune'}, to_coef='1 / hlos * (1 - dh) * (1 - priorinf_fail_rate) * (1-mab_prev)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S'}, to_coef='1 / hlos * (1 - dh) * priorinf_fail_rate * (1-mab_prev)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'immun': 'immune'}, to_coef='1 / (hlos * mab_hlos_adj) * (1 - dh) * (1 - priorinf_fail_rate) * mab_prev')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S'}, to_coef='1 / (hlos * mab_hlos_adj) * (1 - dh) * priorinf_fail_rate * mab_prev')

            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'D'}, to_coef='gamm * dnh * (1 - severe_immunity)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'D'}, to_coef='1 / hlos * dh')

        # immunity decay
        logger.debug(f"{str(self.tags)} Building immunity decay flows")
        for seir in [seir for seir in self.attrs['seir'] if seir != 'D']:
            self.add_flows_from_attrs_to_attrs({'seir': seir, 'immun': 'immune'}, {'immun': 'none'}, to_coef='1 / imm_decay_days')

    def prep(self, rebuild_param_lookups=True, pickle_matrices=True, outdir=None, **build_param_lookup_args):
        logger.info(f"{str(self.tags)} Prepping Model")
        if rebuild_param_lookups:
            self.build_param_lookups(**build_param_lookup_args)
        self.build_ode_flows()
        self.build_region_picker_matrix()
        self.compile()
        # initialize solution dataframe with all NA values
        self.solution_y = np.ndarray(shape=(len(self.trange), len(self.compartments_as_index)))
        if pickle_matrices:
            self.pickle_ode_matrices(outdir)