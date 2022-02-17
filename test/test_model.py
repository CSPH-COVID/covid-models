from unittest import TestCase
import datetime as dt

from covid_model.model import CovidModel


class TestCovidModel(TestCase):
    def setUp(self) -> None:
        self.tslices = [10, 20, 30, 40]
        self.tmax = 50
        self.tc = [0.05, 0.15, 0.25, 0.35, 0.45]

        self.model = CovidModel(end_date=CovidModel.default_start_date + dt.timedelta(days=self.tmax))
        self.model.set_specifications(tslices=self.tslices, tc=self.tc)

    def test_apply_tc(self):
        self.model.apply_tc()
        self.fail()
