import datetime as dt
import json
from collections import OrderedDict
import copy

from covid_model.model import CovidModel


class CovidModelWithFutureVariant(CovidModel):

    attr = OrderedDict({**CovidModel.attr, 'variant': CovidModel.attr['variant'] + ['future']})

    def __init__(self, base_model=None, deepcopy_params=True, future_seed_date=None, **spec_args):
        super().__init__(base_model=base_model, deepcopy_params=deepcopy_params, **spec_args)

        # set future seed
        if future_seed_date is not None:
            future_seed_t = (future_seed_date - self.start_date).days
            self.model_params['future_seed'] = {"tslices": [future_seed_t, future_seed_t + 25], "value": [0, 5, 0]}
        elif 'future_seed':
            self.model_params['future_seed'] = 0

        # # apply attribute multipliers for future variant
        # if variant_attr_mults is not None:
        #     self.attribute_multipliers += variant_attr_mults if isinstance(variant_attr_mults, list) else json.load(open(variant_attr_mults))

    def build_ode(self):
        super().build_ode()
        self.add_flows_by_attr({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'future'}, constant='future_seed')


