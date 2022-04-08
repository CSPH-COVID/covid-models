### Python Standard Library ###
from datetime import date
from argparse import ArgumentParser
### Third Party Imports ###
### Local Imports ###


class ModelSpecsArgumentParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('-sid', '--from_specs', type=int)
        self.add_argument('-tsl', '--tslices', nargs='+', type=int)
        self.add_argument('-tc', '--tc', nargs='+', type=float)
        self.add_argument('-p', '--params', type=str, help="path to parameters file to use")
        self.add_argument('-rp', '--region_params', type=str, help="path to region parameters file to use")
        self.add_argument('-r', '--regions', nargs='+', type=str, help="regions to be run, default is no regions (statewide model)")
        self.add_argument('-rv', '--refresh_actual_vacc', action='store_true')
        self.add_argument('-vpp', '--vacc_proj_params', type=str)
        self.add_argument('-rm', '--refresh_actual_mobility', action='store_true')
        self.add_argument('-mm', '--mobility_mode', type=str, choices=['none', 'population_attached', 'location_attached'])
        self.add_argument('-mpp', '--mobility_proj_params', type=str)
        self.add_argument('-tem', '--timeseries_effect_multipliers', type=str)
        self.add_argument('-vprev', '--variant_prevalence', type=str)
        self.add_argument('-mprev', '--mab_prevalence', type=str)
        self.add_argument('-am', '--attribute_multipliers', type=str)
        self.add_argument('-sd', '--start_date', type=date.fromisoformat, help="format: YYYY-MM-DD")
        self.add_argument('-ed', '--end_date', type=date.fromisoformat, help="format: YYYY-MM-DD")
        self.set_defaults(refresh_actual_vacc=False, refresh_actual_mobility=False, mobility_mode="none", start_date=None, regions="co")

        self.specs_args = self.parse_known_args()[0].__dict__.keys()

    def all_args_as_dict(self, update=None):
        update = update if update is not None else {}
        return {**self.parse_args().__dict__, **update}

    def specs_args_as_dict(self, update=None):
        update = update if update is not None else {}
        return {**{k: v for k, v in self.parse_known_args()[0].__dict__.items() if k in self.specs_args}, **update}
