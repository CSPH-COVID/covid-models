import json
import math
import datetime as dt
import scipy.integrate as spi
import scipy.optimize as spo
import pyswarms as ps
from matplotlib import pyplot as plt
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


if __name__ == '__main__':
    engine = db_engine()

    model = CovidModelWithVariants()
    model.set_param('shot2_per_available', 1)
    model.prep(551, engine=engine, params='input/params.json', attribute_multipliers='input/attribute_multipliers.json')

    fig, ax = plt.subplots()

    model.solve_ode({('S', age, 'shot3', 'none', 'imm3'): n for age, n in model.specifications.group_pops.items()})
    mean_params = model.mean_params_as_df
    mean_params[('S', 'immunity')].plot(label='Booster Immunity vs Wildtype')
    mean_params[('S', 'immunity * (1 - delta_immunity_reduction')].plot(label='Booster Immunity vs Delta')
    mean_params[('S', 'immunity * (1 - omicron_immunity_reduction')].plot(label='Booster Immunity vs Omicron')
    (mean_params[('S', 'immunity')] * mean_params[('S', 'severe_immunity')]).plot(label='Booster Immunity vs Severe Wildtype')
    (mean_params[('S', 'immunity * (1 - delta_immunity_reduction')] * mean_params[('S', 'severe_immunity')]).plot(label='Booster Immunity vs Severe Delta')
    (mean_params[('S', 'immunity * (1 - omicron_immunity_reduction')] * mean_params[('S', 'severe_immunity')]).plot(label='Booster Immunity vs Severe Omicron')

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

    plt.legend(loc='best')
    plt.show()
