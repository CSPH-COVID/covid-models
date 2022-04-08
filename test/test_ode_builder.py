### Python Standard Library ###
from unittest import TestCase
from collections import OrderedDict
### Third Party Imports ###
### Local Imports ###
from covid_model.ode_builder import ODEBuilder


class TestODEBuilder(TestCase):
    def setUp(self) -> None:
        self.attributes = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                                       'age': ['0-19', '20-39', '40-64', '65+'],
                                       'vacc': ['none', 'shot1', 'shot2', 'shot3']})
        self.trange = range(180)
        self.ode_builder = ODEBuilder(attributes=self.attributes, trange=self.trange)

    def test_does_cmpt_have_attrs(self):
        self.assertTrue(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'seir': 'S'}))
        self.assertTrue(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'age': '0-19'}))
        self.assertFalse(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'seir': 'E'}))
        self.assertFalse(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'vacc': 'shot1'}))

    def test_does_cmpt_have_attrs_multiple(self):
        self.assertTrue(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'seir': ['S', 'E']}))
        self.assertTrue(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'age': ['0-19', '65+']}))
        self.assertFalse(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'seir': ['E', 'I']}))
        self.assertFalse(self.ode_builder.does_cmpt_have_attrs(('S', '0-19', 'none'), attrs={'vacc': ['shot1', 'shot2']}))
