from argparse import ArgumentParser


class ModelSpecsCliParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('-sid', '--specs', type=int)
        self.add_argument('-tsl', '--tslices', nargs='+', type=int)
        self.add_argument('-tc', '--tc', nargs='+', type=float)
        self.add_argument('-p', '--params', type=str)
        self.add_argument('-rv', '--refresh_actual_vacc', action='store_true')
        self.add_argument('-vpp', '--vacc_proj_params', type=str)
        self.add_argument('-tem', '--timeseries_effect_multipliers', type=str)
        self.add_argument('-vprev', '--variant_prevalence', type=str)
        self.add_argument('-mprev', '--mab_prevalence', type=str)
        self.add_argument('-am', '--attribute_multipliers', type=str)
        self.set_defaults(refresh_actual_vacc=False)

        self.specs_args = self.parse_known_args()[0].__dict__.keys()

    def all_args_as_dict(self, update=None):
        update = update if update is not None else {}
        return {**self.parse_args().__dict__, **update}

    def specs_args_as_dict(self, update=None):
        update = update if update is not None else {}
        return {**{k: v for k, v in self.parse_known_args()[0].__dict__.items() if k in self.specs_args}, **update}
