import pandas as pd
import json
import datetime as dt
from collections import OrderedDict
from covid_model.model_specs import CovidModelSpecifications
from covid_model.ode_builder import ODEBuilder


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel(ODEBuilder, CovidModelSpecifications):
    attr = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                        'age': ['0-19', '20-39', '40-64', '65+'],
                        'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                        'priorinf': ['none', 'non-omicron', 'omicron'],
                        'variant': ['none', 'alpha', 'delta', 'omicron', 'ba2'],
                        'immun': ['none', 'weak', 'strong']})

    param_attr_names = ('age', 'vacc', 'priorinf', 'variant', 'immun')

    default_end_date = dt.date(2022, 5, 31)

    def __init__(self, base_model=None, deepcopy_params=True, **spec_args):

        # if a base model is provided, use its specifications
        if base_model is not None:
            spec_args['from_specs'] = base_model

        # initiate the parent classes; dates will be set in CovidModelSpecifications.__init__
        CovidModelSpecifications.__init__(self, **spec_args)
        ODEBuilder.__init__(self, base_ode_builder=base_model, deepcopy_params=deepcopy_params, trange=range((self.end_date - self.start_date).days), attributes=self.attr, param_attr_names=self.param_attr_names)
        # the var values for the solution; these get populated when self.solve_seir is run
        self.solution = None
        self.solution_y = None
        self.solution_ydf_full = None

    # a model must be prepped before it can be run; if any params EXCEPT the efs (i.e. TC) change, it must be re-prepped
    def prep(self, rebuild_param_lookups=True):
        if rebuild_param_lookups:
            self.build_param_lookups()
        self.build_ode()
        self.compile()

    def build_param_lookups(self, apply_vaccines=True):
        # set TC
        self.apply_tc()

        # prep general parameters
        for name, val in self.model_params.items():
            if not isinstance(val, dict) or 'tslices' not in val.keys():
                self.set_param_using_age_dict(name, val)
            else:
                for i, (tmin, tmax) in enumerate(zip([self.tmin] + val['tslices'], val['tslices'] + [self.tmax])):
                    v = {a: av[i] for a, av in val['value'].items()} if isinstance(val['value'], dict) else val['value'][i]
                    self.set_param_using_age_dict(name, v, trange=range(tmin, tmax))

        if apply_vaccines:
            vacc_per_available = self.get_vacc_per_available()

            # convert to dictionaries for performance lookup
            vacc_per_available_dict = vacc_per_available.to_dict()

            # set the fail rate and vacc per unvacc rate for each dose
            vacc_delay = 14
            for shot in self.attr['vacc'][1:]:
                self.set_param(f'{shot}_per_available', 0, trange=range(0, vacc_delay))
                for age in self.attr['age']:
                    for t in range(vacc_delay, self.tmax):
                        self.set_param(f'{shot}_per_available', vacc_per_available_dict[shot][(t - vacc_delay, age)],
                                       {'age': age}, trange=[t])

        # alter parameters based on timeseries effects
        multipliers = self.get_timeseries_effect_multipliers()
        for param, mult_by_t in multipliers.to_dict().items():
            for t, mult in mult_by_t.items():
                if t in self.trange:
                    self.set_param(param, mult=mult, trange=[t])

        # alter parameters based on attribute multipliers
        if self.attribute_multipliers:
            for attr_mult_specs in self.attribute_multipliers:
                self.set_param(**attr_mult_specs)

    # handy properties for the beginning t, end t, and the full range of t values
    @property
    def tmin(self): return 0

    @property
    def tmax(self): return (self.end_date - self.start_date).days

    @property
    def daterange(self): return pd.date_range(self.start_date, periods=len(self.trange))

    # new exposures by day by group
    @property
    def new_infections(self):
        return self.solution_sum('seir')['E'] / self.model_params['alpha']

    # estimated reproduction number (length of infection * new_exposures / current_infections
    @property
    def re_estimates(self):
        infect_duration = 1 / self.model_params['gamm']
        infected = (self.solution_sum('seir')['I'].shift(3) + self.solution_sum('seir')['A'].shift(3))
        return infect_duration * self.new_infections.groupby('t').sum() / infected

    # provide a dictionary by age (as in params.json) and update parameters accordingly
    def set_param_using_age_dict(self, name, val, trange=None):
        if not isinstance(val, dict):
            self.set_param(name, val, trange=trange)
        else:
            for age, v in val.items():
                self.set_param(name, v, attrs={'age': age}, trange=trange)

    # set TC by slice, and update non-linear multipliers; defaults to reseting the last TC values
    def apply_tc(self, tc=None, tslices=None, suppress_ode_rebuild=False):
        # if tslices are provided, replace any tslices >= tslices[0] with the new tslices
        if tslices is not None:
            self.tslices = [tslice for tslice in self.tslices if tslice < tslices[0]] + tslices
            self.tc = self.tc[:len(self.tslices) + 1]  # truncate tc if longer than tslices
            self.tc += [self.tc[-1]] * (1 + len(self.tslices) - len(self.tc))  # extend tc if shorter than tslices

        # if tc is provided, replace the
        if tc is not None:
            self.tc = self.tc[:-len(tc)] + tc

        # if the lengths do not match, raise an error
        if len(self.tc) != len(self.tslices) + 1:
            raise ValueError(f'The length of tc ({len(self.tc)}) must be equal to the length of tslices ({len(self.tslices)}) + 1.')

        # apply to the ef parameter
        # TODO: the ODE is no longer using this (uses non-linear multiplier instead), so we should check if this is used and get rid of it
        for tmin, tmax, tc in zip([self.tmin] + self.tslices, self.tslices + [self.tmax], self.tc):
            self.set_param('ef', tc, trange=range(tmin, tmax))

        # apply the new TC values to the non-linear multiplier to update the ODE
        # TODO: only update the nonlinear multipliers for TCs that have been changed
        if not suppress_ode_rebuild:
            for tmin, tmax, tc in zip([self.tmin] + self.tslices, self.tslices + [self.tmax], self.tc):
                self.set_nonlinear_multiplier(1 - tc, trange=range(tmin, tmax))

    # build ODE
    def build_ode(self):
        self.reset_ode()

        # tc
        self.apply_tc()

        # vaccination
        # first shot
        self.add_flows_by_attr({'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'weak'}, coef=f'shot1_per_available * (1 - shot1_fail_rate)')
        self.add_flows_by_attr({'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'none'}, coef=f'shot1_per_available * shot1_fail_rate')
        # second and third shot
        for i in [2, 3]:
            for immun in self.attributes['immun']:
                # if immun is none, that means that the first vacc shot failed, which means that future shots may fail as well
                if immun == 'none':
                    self.add_flows_by_attr({'vacc': f'shot{i-1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, coef=f'shot{i}_per_available * (1 - shot{i}_fail_rate / shot{i-1}_fail_rate)')
                    self.add_flows_by_attr({'vacc': f'shot{i-1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'none'}, coef=f'shot{i}_per_available * (shot{i}_fail_rate / shot{i-1}_fail_rate)')
                else:
                    self.add_flows_by_attr({'vacc': f'shot{i-1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, coef=f'shot{i}_per_available')

        # seed variants
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'none'}, constant='initial_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'alpha'}, constant='alpha_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'delta'}, constant='delta_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'omicron'}, constant='om_seed')
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'ba2'}, constant='ba2_seed')

        # exposure
        asymptomatic_transmission = '(1 - immunity) * betta / total_pop'
        for variant in self.attributes['variant']:
            sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant})
            asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant})
            self.add_flows_by_attr({'seir': 'S', 'variant': 'none'}, {'seir': 'E', 'variant': variant}, coef=f'lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
            self.add_flows_by_attr({'seir': 'S', 'variant': 'none'}, {'seir': 'E', 'variant': variant}, coef=asymptomatic_transmission, scale_by_cmpts=asympt_cmpts)

        # disease progression
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'I'}, coef='1 / alpha * pS')
        self.add_flows_by_attr({'seir': 'E'}, {'seir': 'A'}, coef='1 / alpha * (1 - pS)')
        self.add_flows_by_attr({'seir': 'I'}, {'seir': 'Ih'}, coef='gamm * hosp * (1 - severe_immunity)')

        # disease termination
        for variant in self.attributes['variant']:
            # TODO: Rename "non-omicron" to "other"; will need to make the change in attribute_multipliers, which will break old specifications
            priorinf = variant if variant != 'none' and variant in self.attributes['priorinf'] else 'non-omicron'
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='gamm * (1 - hosp - dnh) * (1 - priorinf_fail_rate)')
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * (1 - hosp - dnh) * priorinf_fail_rate')
            self.add_flows_by_attr({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='gamm * (1 - priorinf_fail_rate)')
            self.add_flows_by_attr({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * priorinf_fail_rate')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='1 / hlos * (1 - dh) * (1 - priorinf_fail_rate)')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * (1 - dh) * priorinf_fail_rate')
            self.add_flows_by_attr({'seir': 'I', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * dnh * (1 - severe_immunity)')
            self.add_flows_by_attr({'seir': 'Ih', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * dh')

        # immunity decay
        # self.add_flows_by_attr({'immun': 'imm3'}, {'immun': 'imm2'}, coef='1 / imm3_decay_days')
        # self.add_flows_by_attr({'immun': 'imm2'}, {'immun': 'imm1'}, coef='1 / imm2_decay_days')
        # self.add_flows_by_attr({'immun': 'imm1'}, {'immun': 'imm0'}, coef='1 / imm1_decay_days')
        self.add_flows_by_attr({'immun': 'strong'}, {'immun': 'weak'}, coef='1 / imm_decay_days')

    # define initial state y0
    @property
    def y0_dict(self):
        y0d = {('S', age, 'none', 'none', 'none', 'none'): n for age, n in self.group_pops.items()}
        return y0d

    # override solve_ode to use default y0_dict
    def solve_seir(self, method='RK45', y0_dict=None):
        y0_dict = y0_dict if y0_dict is not None else self.y0_dict
        self.solve_ode(y0_dict=y0_dict, method=method)

    # count the total hosps by t as the sum of Ih and Ic
    def total_hosps(self):
        return self.solution_sum('seir')['Ih']

    # count the new exposed individuals by day
    def new_exposed(self):
        sum_df = self.solution_sum('seir')
        return sum_df['E'] - sum_df['E'].shift(1) + sum_df['E'].shift(1) / self.model_params['alpha']

    # immunity
    def immunity(self, variant='omicron', vacc_only=False, to_hosp=False):
        params = self.params_as_df
        group_by_attr_names = [attr_name for attr_name in self.param_attr_names if attr_name != 'variant']
        n = self.solution_sum(group_by_attr_names).stack(level=group_by_attr_names)

        if vacc_only:
            params.loc[params.index.get_level_values('vacc') == 'none', 'immunity'] = 0
            params.loc[params.index.get_level_values('vacc') == 'none', 'severe_immunity'] = 0

        variant_params = params.xs(variant, level='variant')
        if to_hosp:
            weights = variant_params['hosp'] * n
            return (weights * (1 - (1 - variant_params['immunity']) * (1 - variant_params['severe_immunity']))).groupby('t').sum() / weights.groupby('t').sum()
        else:
            return (n * variant_params['immunity']).groupby('t').sum() / n.groupby('t').sum()

    # write to covid_model.results
    def write_results_to_db(self, engine=None, new_spec=False, vals_json_attr='seir', cmpts_json_attrs=('age', 'vacc'), sim_id=None, sim_result_id=None):

        # if there's no existing fit assigned, create a new fit and assign that one
        if self.spec_id is None or new_spec:
            self.write_to_db(engine)

        # build data frame with index of (t, age, vacc) and one column per seir cmpt
        solution_sum_df = self.solution_sum([vals_json_attr] + list(cmpts_json_attrs)).stack(cmpts_json_attrs)

        # Merge R and R2 into one column
        if 'R2' in solution_sum_df.columns:
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
            df['spec_id'] = self.spec_id
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
