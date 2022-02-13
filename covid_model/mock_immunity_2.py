import json
import math
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import pyswarms as ps
from matplotlib import pyplot as plt, ticker as mtick
import matplotlib as mpl
from cycler import cycler
from sqlalchemy import MetaData
from datetime import datetime
import itertools
from collections import OrderedDict

from covid_model.analysis.charts import modeled, actual_hosps
from covid_model.data_imports import ExternalHosps, ExternalVaccWithProjections
from covid_model.db import db_engine
from covid_model.model_specs import CovidModelSpecifications
from covid_model.utils import *
from covid_model.ode_builder import *
from model_with_immunity_rework import CovidModelWithVariants

#
# # class used to run the model given a set of parameters, including transmission control (ef)
# class ImmunityModel(ODEBuilder):
#     attr = OrderedDict({'variant': ['none', 'wt', 'omicron'],
#                         'immun': ['none', 'imm1', 'imm2', 'imm3']})
#
#     param_attr_names = ('variant', 'immun')
#
#     # # the starting date of the model
#     # default_start_date = dt.datetime(2020, 1, 24)
#     #
#     # # def __init__(self, tslices=None, efs=None, fit_id=None, engine=None, **ode_builder_args):
#     # def __init__(self, start_date=dt.date(2020, 1, 24), end_date=dt.date(2022, 5, 31), **ode_builder_args):
#     #     self.start_date = start_date
#     #     self.end_date = end_date
#
#     # build ODE
#     def build_ode(self):
#         self.reset_ode()
#         self.add_flows_by_attr()


# class ImmunityModel(CovidModelWithVariants):
#
#     @property
#     def y0_dict(self):
#         y0d = {('S', age, 'shot3', 'none', 'imm3'): n for age, n in self.specifications.group_pops.items()}
#         # y0d[('I', '40-64', 'none', 'wt', 'none')] = 2.2
#         # y0d[('S', '40-64', 'none', 'none', 'none')] -= 2.2
#         return y0d

    # # don't apply vaccinations at all
    # def prep(self, specs=None, **specs_args):
    #     self.set_specifications(specs=specs, **specs_args)
    #     self.apply_specifications(apply_vaccines=False)
    #     self.build_ode()


def build_default_model():
    model = CovidModelWithVariants(end_date=dt.date(2021, 1, 24))
    model.set_specifications(702, engine=engine, params='input/params.json', attribute_multipliers='input/attribute_multipliers.json')
    model.apply_specifications(apply_vaccines=False)
    # model.apply_new_vacc_params()
    model.set_param('shot1_per_available', 0)
    model.set_param('shot2_per_available', 0)
    model.set_param('shot3_per_available', 0)
    model.set_param('ef', 1)
    model.set_param('betta', 0)

    return model
    # model.build_ode()
    # model.compile()

    # model.prep(551, engine=engine, params='input/params.json', attribute_multipliers='input/attribute_multipliers.json')

if __name__ == '__main__':
    engine = db_engine()

    variants = {
        'Non-Omicron': 'none',
        'Omicron': 'omicron'
    }

    immunities = {
        'Dose-2': {'initial_attrs': {'seir': 'S'}, 'params': {f'shot{i}_per_available': 1 for i in [1, 2]}},
        'Booster': {'initial_attrs': {'seir': 'S'}, 'params': {f'shot{i}_per_available': 1 for i in [1, 2, 3]}},
        'Prior Delta Infection': {'initial_attrs': {'seir': 'E', 'variant': 'none'}, 'params': {}},
        'Prior Omicron Infection': {'initial_attrs': {'seir': 'E', 'variant': 'omicron'}, 'params': {}}
    }

    fig, axs = plt.subplots(2, 2)
    # colors = ['violet', 'mediumorchid', 'indigo', 'gold', 'orange']
    # colors = ['violet', 'indigo', 'gold', 'orange']
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['violet', 'indigo', 'gold', 'orange'])

    for (immunity_label, immunity_specs), ax in zip(immunities.items(), axs.flatten()):
        ax.set_prop_cycle(cycler(color=['paleturquoise', 'darkcyan', 'violet', 'indigo']))
        ax.set_title(f'Immunity from {immunity_label}')

        print(f'Prepping and running model for {immunity_label} immunity...')
        model = build_default_model()
        for k, v in immunity_specs['params'].items():
            model.set_param(k, v)
        model.build_ode()
        model.compile()
        model.solve_ode({model.get_default_cmpt_by_attrs({**immunity_specs['initial_attrs'], 'age': age}): n for age, n in model.specifications.group_pops.items()})

        params = model.params_as_df
        group_by_attr_names = ['seir'] + [attr_name for attr_name in model.param_attr_names if attr_name != 'variant']
        n = model.solution_sum(group_by_attr_names).stack(level=group_by_attr_names).xs('S', level='seir')

        for variant_label, variant in variants.items():
            if immunity_label != 'Prior Omicron Infection' or variant_label == 'Non-Omicron':
                variant_params = params.xs(variant, level='variant')

                net_severe_immunity = (n * (1 - (1 - variant_params['immunity']) * (1 - variant_params['severe_immunity']))).groupby('t').sum() / n.groupby('t').sum()
                net_severe_immunity.plot(label=f'Immunity vs Severe {"Disease" if immunity_label == "Prior Omicron Infection" else variant_label}', ax=ax)

                immunity = (n * variant_params['immunity']).groupby('t').sum() / n.groupby('t').sum()
                immunity.plot(label=f'Immunity vs {"Infection" if immunity_label == "Prior Omicron Infection" else variant_label}', ax=ax)

        ax.legend(loc='best')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(color='lightgray')
        ax.set_xlabel(f'Days Since {immunity_label}')
        ax.set_ylim((0, 1))
        ax.set_xticks(np.arange(0, 365, 30))
        ax.set_xlim((30, 360))

    # plt.legend(loc='best')
    fig.tight_layout()
    plt.show()


    # print(immunity)
    # immunity.xs('none', level='variant').plot(label='2-Dose Immunity vs Wildtype')
    # immunity.xs('omicron', level='variant').plot(label='2-Dose Immunity vs Omicron')
    # net_severe_immunity.xs('none', level='variant').plot(label='2-Dose Immunity vs Severe Wildtype')
    # net_severe_immunity.xs('omicron', level='variant').plot(label='2-Dose Immunity vs Severe Omicron')
    # immunity_to_wt.plot(label='2-Dose Immunity vs Wildtype')

    # mean_params = model.mean_params_as_df(['seir', 'variant'])
    # # print(mean_params.xs('immunity', level='param', axis=1))
    # immunity = mean_params.xs('immunity', level='param', axis=1)#.xs('S', level='seir', axis=1)
    # print(immunity)
    # exit()
    # severe_immunity = mean_params.xs('severe_immunity', level='param', axis=1).xs('I', level='seir', axis=1)
    # net_severe_immunity = 1 - (1 - immunity) * (1 - severe_immunity)

    # immunity['none'].plot(label='Booster Immunity vs Wildtype')
    # immunity['omicron'].plot(label='Booster Immunity vs Omicron')
    # severe_immunity['none'].plot(label='Booster Immunity vs Severe Wildtype')
    # severe_immunity['omicron'].plot(label='Booster Immunity vs Severe Omicron')


    # mean_params[('S', 'immunity * (1 - delta_immunity_reduction')].plot(label='Booster Immunity vs Delta')
    # mean_params[('S', 'immunity * (1 - omicron_immunity_reduction')].plot(label='Booster Immunity vs Omicron')
    # (mean_params[('S', 'immunity')] * mean_params[('S', 'severe_immunity')]).plot(label='Booster Immunity vs Severe Wildtype')
    # (mean_params[('S', 'immunity * (1 - delta_immunity_reduction')] * mean_params[('S', 'severe_immunity')]).plot(label='Booster Immunity vs Severe Delta')
    # (mean_params[('S', 'immunity * (1 - omicron_immunity_reduction')] * mean_params[('S', 'severe_immunity')]).plot(label='Booster Immunity vs Severe Omicron')

    # model.solve_ode({('S', age, 'shot3', 'omicron', 'imm3'): n for age, n in model.specifications.group_pops.items()})
    # mean_params = model.mean_params_as_df
    # mean_params[('S', 'immunity')].plot(label='Booster Immunity vs Omicron')
    # mean_params[('S', 'severe_immunity')].plot(label='Booster Immunity vs Severe Omicron')


    # by_immun = model.solution_sum('immun')
    # ((by_immun * model.params_as_df.loc[(0, '0-19', 'shot2', 'wt'), 'immunity']).sum(axis=1) / by_immun.sum(axis=1)).plot(label='Booster Immunity vs Wildtype')
    # ((by_immun * model.params_as_df.loc[(0, '0-19', 'shot2', 'omicron'), 'immunity']).sum(axis=1) / by_immun.sum(axis=1)).plot(label='Booster Immunity vs Omicron')
    #
    #
    # ((by_immun * model.params_as_df.loc[(0, '0-19', 'shot2', 'wt'), 'immunity']).sum(axis=1) / by_immun.sum(axis=1)).plot(label='Booster Immunity vs Wildtype')
    # ((by_immun * model.params_as_df.loc[(0, '0-19', 'shot2', 'omicron'), 'immunity']).sum(axis=1) / by_immun.sum(axis=1)).plot(label='Booster Immunity vs Omicron')


