import json
import math
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import pyswarms as ps
from sqlalchemy import MetaData
from datetime import datetime
import itertools
from collections import OrderedDict
from covid_model.data_imports import ExternalHosps, ExternalVaccWithProjections
from covid_model.model_specs import CovidModelSpecifications
from covid_model.utils import *
from covid_model.ode_builder import *


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel(ODEBuilder):
    attr = OrderedDict({'seir': ['S', 'E', 'I', 'Ih', 'A', 'R', 'R2', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['unvacc', 'vacc', 'vacc_fail']})

    param_attr_names = ('age', 'vacc')

    # the starting date of the model
    default_start_date = dt.datetime(2020, 1, 24)

    # def __init__(self, tslices=None, efs=None, fit_id=None, engine=None, **ode_builder_args):
    def __init__(self, start_date=dt.date(2020, 1, 24), end_date=dt.date(2022, 5, 31), **ode_builder_args):
        self.start_date = start_date
        self.end_date = end_date

        ODEBuilder.__init__(self, trange=range((self.end_date - self.start_date).days), attributes=self.attr, param_attr_names=self.param_attr_names)

        self.specifications = None

        # transmission control parameters
        # self.efs = efs
        # self.ef_by_t = None

        # the var values for the solution; these get populated when self.solve_seir is run
        self.solution = None
        self.solution_y = None
        self.solution_ydf_full = None

    def set_specifications(self, specs=None, engine=None,
                           tslices=None, tc=None, params=None,
                           refresh_actual_vacc=False, vacc_proj_params=None, vacc_immun_params=None,
                           timeseries_effect_multipliers=None, variant_prevalence=None, mab_prevalence=None,
                           attribute_multipliers=None):

        if specs is not None:
            if not isinstance(specs, (int, np.int64)):
                self.specifications = specs
            else:
                self.specifications = CovidModelSpecifications.from_db(engine, specs, new_end_date=self.end_date)

        if self.specifications is None:
            self.specifications = CovidModelSpecifications(start_date=self.start_date, end_date=self.end_date)

        if tslices or tc:
            self.specifications.set_tc(tslices, tc)
        if params:
            self.specifications.set_model_params(params)
        if refresh_actual_vacc:
            self.specifications.set_actual_vacc(engine)
        if refresh_actual_vacc or vacc_proj_params:
            self.specifications.set_vacc_proj(vacc_proj_params)
        if vacc_immun_params:
            self.specifications.set_vacc_immun(vacc_immun_params)

        # TODO: make add_timeseries_effect use existing prevalence and existing multipliers if new prevalence or mults are not provided
        # if variant_prevalence or param_multipliers:
        if variant_prevalence:
            self.specifications.add_timeseries_effect('variant', prevalence_data=variant_prevalence, param_multipliers=timeseries_effect_multipliers, fill_forward=True)
        # if mab_prevalence or param_multipliers:
        if mab_prevalence:
            self.specifications.add_timeseries_effect('mab', prevalence_data=mab_prevalence, param_multipliers=timeseries_effect_multipliers, fill_forward=True)

        if attribute_multipliers:
            self.specifications.set_attr_mults(attribute_multipliers)

    # a model must be prepped before it can be run; if any params EXCEPT the efs (i.e. TC) change, it must be re-prepped
    def prep(self, specs=None, **specs_args):

        self.set_specifications(specs=specs, **specs_args)
        self.apply_specifications()
        self.build_ode()
        # self.compile()

    def apply_specifications(self, specs: CovidModelSpecifications = None, apply_vaccines=True):
        if specs is not None:
            self.specifications = specs

        # set TC
        self.apply_tc(suppress_ode_rebuild=True)

        # prep general parameters
        for name, val in self.specifications.model_params.items():
            if not isinstance(val, dict) or 'tslices' not in val.keys():
                self.set_param_using_age_dict(name, val)
            else:
                for i, (tmin, tmax) in enumerate(zip([self.tmin] + val['tslices'], val['tslices'] + [self.tmax])):
                    v = {a: av[i] for a, av in val['value'].items()} if isinstance(val['value'], dict) else val['value'][i]
                    self.set_param_using_age_dict(name, v, trange=range(tmin, tmax))

        if apply_vaccines:
            # get vacc rates and efficacy from specifications
            vacc_per_unvacc = self.specifications.get_vacc_rate_per_unvacc()
            vacc_mean_efficacy = self.specifications.get_vacc_mean_efficacy()
            vacc_mean_efficacy_vs_delta = self.specifications.get_vacc_mean_efficacy(k='delta_vacc_eff_k')
            vacc_fail_per_vacc = self.specifications.get_vacc_fail_per_vacc()
            vacc_fail_reduction_per_fail = self.specifications.get_vacc_fail_reduction_per_vacc_fail()

            # convert to dictionaries for performance lookup
            vacc_per_unvacc_dict = vacc_per_unvacc.to_dict()
            vacc_mean_efficacy_dict = vacc_mean_efficacy.to_dict()
            vacc_mean_efficacy_vs_delta_dict = vacc_mean_efficacy_vs_delta.to_dict()
            vacc_fail_reduction_per_fail_dict = vacc_fail_reduction_per_fail.to_dict()

            # set the fail rate and vacc per unvacc rate for each dose
            for shot in vacc_per_unvacc.columns:
                for age in self.attr['age']:
                    self.set_param(f'{shot}_fail_rate', vacc_fail_per_vacc[shot][age], {'age': age})
                    for t in self.trange:
                        self.set_param(f'{shot}_per_unvacc', vacc_per_unvacc_dict[shot][(t, age)], {'age': age}, trange=[t])

            # set hospitalization and mortality to 0 among the vacc successes
            self.set_param('hosp', 0, attrs={'vacc': 'vacc'})
            self.set_param('dnh', 0, attrs={'vacc': 'vacc'})

            # set vacc efficacy and fail reduction over time
            self.set_param('vacc_eff', 0, {'vacc': 'unvacc'})
            self.set_param('vacc_eff', 0, {'vacc': 'vacc_fail'})
            self.set_param('vacc_eff_vs_delta', 0, {'vacc': 'unvacc'})
            self.set_param('vacc_eff_vs_delta', 0, {'vacc': 'vacc_fail'})
            for age in self.attr['age']:
                for t in self.trange:
                    self.set_param('vacc_eff', vacc_mean_efficacy_dict[(t, age)], {'age': age, 'vacc': 'vacc'}, trange=[t])
                    self.set_param('vacc_eff_vs_delta', vacc_mean_efficacy_vs_delta_dict[(t, age)], {'age': age, 'vacc': 'vacc'}, trange=[t])
                    self.set_param('vacc_fail_reduction_per_vacc_fail', vacc_fail_reduction_per_fail_dict[(t, age)], {'age': age, 'vacc': 'vacc'}, trange=[t])

        # alter parameters based on timeseries effects
        multipliers = self.specifications.get_timeseries_effect_multipliers()
        for param, mult_by_t in multipliers.to_dict().items():
            for t, mult in mult_by_t.items():
                if t in self.trange:
                    self.set_param(param, mult=mult, trange=[t])

        # alter parameters based on attribute multipliers
        if self.specifications.attr_mults:
            for attr_mult_specs in self.specifications.attr_mults:
                self.set_param(**attr_mult_specs)

    # handy properties for the beginning t, end t, and the full range of t values
    @property
    # def tmin(self): return self.tslices[0]
    def tmin(self): return 0

    @property
    # def tmax(self): return self.tslices[-1]
    def tmax(self): return (self.end_date - self.start_date).days

    @property
    def daterange(self): return pd.date_range(self.start_date, periods=len(self.trange))

    # new exposures by day by group
    @property
    def new_infections(self):
        return self.solution_sum('seir')['E'] / self.specifications.model_params['alpha']

    # estimated reproduction number (length of infection * new_exposures / current_infections
    @property
    def re_estimates(self):
        infect_duration = 1 / self.specifications.model_params['gamm']
        infected = (self.solution_sum('seir')['I'].shift(3) + self.solution_sum('seir')['A'].shift(3))
        return infect_duration * self.new_infections.groupby('t').sum() / infected

    def set_param_using_age_dict(self, name, val, trange=None):
        if not isinstance(val, dict):
            self.set_param(name, val, trange=trange)
        else:
            for age, v in val.items():
                self.set_param(name, v, attrs={'age': age}, trange=trange)

    # set ef by slice and lookup dicts
    def apply_tc(self, tc=None, tslices=None, suppress_ode_rebuild=False):
        if tslices is not None:
            self.specifications.tslices = tslices
        if tc is not None:
            self.specifications.tc = tc

        if len(self.specifications.tc) != len(self.specifications.tslices) + 1:
            raise ValueError(f'The length of tc ({len(self.specifications.tc)}) must be equal to the length of tslices ({len(self.specifications.tslices)}) + 1.')

        for tmin, tmax, ef in zip([self.tmin] + self.specifications.tslices, self.specifications.tslices + [self.tmax], self.specifications.tc):
            self.set_param('ef', ef, trange=range(tmin, tmax))

        # the ODE needs to be rebuilt with the new TC values
        # it would be good to refactor the ODEBuilder to support fittable parameters that are easier to adjust cheaply
        if len(self.terms) > 0 and not suppress_ode_rebuild:
            self.rebuild_ode_with_new_tc()

    # build ODE
    def build_ode(self):
        self.reset_ode()
        self.build_SR_to_E_ode()
        for age in self.attributes['age']:
            for seir in self.attributes['seir']:
                self.add_flow((seir, age, 'unvacc'), (seir, age, 'vacc_fail'), 'shot1_per_unvacc * shot1_fail_rate')
                self.add_flow((seir, age, 'unvacc'), (seir, age, 'vacc'), 'shot1_per_unvacc * (1 - shot1_fail_rate)')
                self.add_flow((seir, age, 'vacc_fail'), (seir, age, 'vacc'), 'vacc_fail_reduction_per_vacc_fail')
            for vacc in self.attributes['vacc']:
                self.add_flow(('E', age, vacc), ('I', age, vacc), '1 / alpha * pS')
                self.add_flow(('E', age, vacc), ('A', age, vacc), '1 / alpha * (1 - pS)')
                self.add_flow(('I', age, vacc), ('Ih', age, vacc), 'gamm * hosp')
                self.add_flow(('I', age, vacc), ('D', age, vacc), 'gamm * dnh')
                self.add_flow(('I', age, vacc), ('R', age, vacc), 'gamm * (1 - hosp - dnh) * immune_rate_I')
                self.add_flow(('I', age, vacc), ('S', age, vacc), 'gamm * (1 - hosp - dnh) * (1 - immune_rate_I)')
                self.add_flow(('A', age, vacc), ('R', age, vacc), 'gamm * immune_rate_A')
                self.add_flow(('A', age, vacc), ('S', age, vacc), 'gamm * (1 - immune_rate_A)')
                self.add_flow(('Ih', age, vacc), ('D', age, vacc), '1 / hlos * dh')
                self.add_flow(('Ih', age, vacc), ('R', age, vacc), '1 / hlos * (1 - dh) * immune_rate_I')
                self.add_flow(('Ih', age, vacc), ('S', age, vacc), '1 / hlos * (1 - dh) * (1 - immune_rate_I)')
                self.add_flow(('R', age, vacc), ('R2', age, vacc), '1 / immune_decay_days_1')
                self.add_flow(('R2', age, vacc), ('S', age, vacc), '1 / immune_decay_days_2')

    def build_SR_to_E_ode(self):
        self.reset_terms({'seir': 'S'}, {'seir': 'E'})
        vacc_eff_w_delta = '(vacc_eff * nondelta_prevalence + vacc_eff_vs_delta * (1 - nondelta_prevalence))'
        base_transm = f'betta * (1 - ef) * (1 - {vacc_eff_w_delta}) / total_pop'
        infectious_cmpts = [(s, a, v) for a in self.attributes['age'] for v in self.attributes['vacc'] for s in ['I', 'A']]
        infectious_cmpt_coefs = [' * '.join(['lamb' if seir == 'I' else '1']) for seir, age, vacc in infectious_cmpts]
        for age in self.attributes['age']:
            self.add_flow(('S', age, 'unvacc'), ('E', age, 'unvacc'), base_transm, scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
            self.add_flow(('S', age, 'vacc'), ('E', age, 'vacc'), base_transm, scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)
            self.add_flow(('S', age, 'vacc_fail'), ('E', age, 'vacc_fail'), base_transm, scale_by_cmpts=infectious_cmpts, scale_by_cmpts_coef=infectious_cmpt_coefs)

    # reset terms that depend on TC; this takes about 0.08 sec, while rebuilding the whole ODE takes ~0.90 sec
    def rebuild_ode_with_new_tc(self):
        self.reset_terms({'seir': 'S'}, {'seir': 'E'})
        self.build_SR_to_E_ode()

    # define initial state y0
    @property
    def y0_dict(self):
        y0d = {('S', age, 'unvacc'): n for age, n in self.specifications.group_pops.items()}
        y0d[('I', '40-64', 'unvacc')] = 2.2
        y0d[('S', '40-64', 'unvacc')] -= 2.2
        return y0d

    # override solve_ode to use default y0_dict
    def solve_seir(self, method='RK45'):
        self.solve_ode(y0_dict=self.y0_dict, method=method)

    # count the total hosps by t as the sum of Ih and Ic
    def total_hosps(self):
        return self.solution_sum('seir')['Ih']

    # count the new exposed individuals by day
    def new_exposed(self):
        sum_df = self.solution_sum('seir')
        return sum_df['E'] - sum_df['E'].shift(1) + sum_df['E'].shift(1) / self.specifications.model_params['alpha']

    # immunity
    def immunity(self, variant='omicron', vacc_only=False, to_hosp=False):
        susc_by_t = {t: 0 for t in self.trange}
        for from_cmpt in self.compartments:
            n = self.solution_ydf[from_cmpt]
            for to_cmpt in self.filter_cmpts_by_attrs({'seir': 'E', 'variant': variant}):
                # we only want actual flows into E, not movement within E
                if from_cmpt[0] != 'E' or (to_cmpt[2] == from_cmpt[2] and vacc_only):
                    terms = self.get_terms_by_cmpt(('S', ) + from_cmpt[1:3] + ('none', ) if vacc_only else from_cmpt, to_cmpt)
                    for term in terms:
                        if term.coef_by_t is not None:  # if the term is a constant, coef_by_t will be None
                            for t in self.trange:
                                susc_rate = term.coef_by_t[t] * self.params[t][to_cmpt[1:]]['total_pop'] / self.params[t][to_cmpt[1:]]['betta'] / (1 - self.params[t][to_cmpt[1:]]['ef'])
                                susc_by_t[t] += n[t] * susc_rate

        susc = pd.Series(susc_by_t)

        if to_hosp:
            params_df = self.params_as_df
            hosp_vuln = params_df['hosp'] / params_df['hosp'].xs('unvacc', level='vacc')
            infected = self.solution_ydf.transpose().xs('I', level='seir') + self.solution_ydf.transpose().xs('A', level='seir')
            infected = infected.stack().reorder_levels(hosp_vuln.index.names)
            susc *= (infected * hosp_vuln).groupby('t').sum() / infected.groupby('t').sum()

        return 1 - susc / self.specifications.model_params['total_pop']

    # write to covid_model.results in Postgres
    def write_to_db(self, engine=None, new_spec=False, vals_json_attr='seir', cmpts_json_attrs=('age', 'vacc'), sim_id=None, sim_result_id=None):

        # if there's no existing fit assigned, create a new fit and assign that one
        if self.specifications.spec_id is None or new_spec:
            self.specifications.write_to_db(engine)

        # build data frame with index of (t, age, vacc) and one column per seir cmpt
        solution_sum_df = self.solution_sum([vals_json_attr] + list(cmpts_json_attrs)).stack(cmpts_json_attrs)

        # Merge R and R2 into one column
        solution_sum_df['R'] += solution_sum_df['R2']
        del solution_sum_df['R2']

        # build unique parameters dataframe
        params_df = self.params_as_df
        grouped = params_df.groupby(['t'] + list(cmpts_json_attrs))
        unique_params = [param for param, is_unique in (grouped.nunique() == 1).all().iteritems() if is_unique]
        unique_params_df = grouped.max()[unique_params]

        # build export dataframe
        df = pd.DataFrame(index=solution_sum_df.index)
        df['t'] = solution_sum_df.index.get_level_values('t')
        df['cmpt'] = solution_sum_df.index.droplevel('t').to_frame().to_dict(orient='records') if solution_sum_df.index.nlevels > 1 else None
        df['vals'] = solution_sum_df.to_dict(orient='records')
        for col in ['cmpt', 'vals']:
            df[col] = df[col].map(lambda d: json.dumps(d, ensure_ascii=False))

        # if a sim_id is provided, insert it as a simulation result; some fields are different
        if sim_id is None:
            table = 'results_v2'
            df['spec_id'] = self.specifications.spec_id
            df['result_id'] = pd.read_sql(f'select coalesce(max(result_id), 0) from covid_model.{table}', con=engine).values[0][0] + 1
            df['created_at'] = dt.datetime.now()
            df['params'] = unique_params_df.apply(lambda x: json.dumps(x.to_dict(), ensure_ascii=False), axis=1)
        else:
            table = 'simulation_results_v2'
            df['sim_id'] = sim_id
            df['sim_result_id'] = sim_result_id
            df['tc'] = unique_params_df['ef']

        # write to database
        results = df.to_sql(table
                  , con=engine, schema='covid_model'
                  , index=False, if_exists='append', method='multi', chunksize=1000000)

    def write_gparams_lookup_to_csv(self, fname):
        df_by_t = {t: pd.DataFrame.from_dict(df_by_group, orient='index') for t, df_by_group in self.params.items()}
        pd.concat(df_by_t, names=['t', 'age']).to_csv(fname)
