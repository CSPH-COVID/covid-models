### Python Standard Library ###
from unittest import TestCase
import datetime as dt
### Third Party Imports ###
### Local Imports ###
from covid_model.model import CovidModel


class TestCovidModelApplyTC(TestCase):
    def setUp(self) -> None:
        self.tslices = [10, 20, 30, 40]
        self.tmax = 50
        self.tc = [0.05, 0.15, 0.25, 0.35, 0.45]

        self.model = CovidModel(end_date=CovidModel.default_start_date + dt.timedelta(days=self.tmax))
        self.model.set_specifications(tslices=self.tslices, tc=self.tc)

    # tc-only
    def test_apply_tc_replace_tc(self):
        new_tc = [1.05, 1.15, 1.25, 1.35, 1.45]
        self.model.update_tc(tc=new_tc)
        self.assertEqual(self.model.tc, new_tc)

    def test_apply_tc_replace_tc_partial(self):
        new_tc = [1.25, 1.35, 1.45]
        self.model.update_tc(tc=new_tc)
        self.assertEqual(self.model.tc, self.tc[:2] + new_tc)

    # tslices-only
    def test_apply_tc_replace_tslices(self):
        new_tslices = [10, 120, 130, 140]
        self.model.update_tc(tslices=new_tslices)
        self.assertEqual(self.model.tc_tslices, new_tslices)

    def test_apply_tc_replace_tslices_partial(self):
        self.model.update_tc(tslices=[20, 130, 140])
        self.assertEqual(self.model.tc_tslices, [10, 20, 130, 140])

    def test_apply_tc_append_tslices(self):
        self.model.update_tc(tslices=[41, 42])
        self.assertEqual(self.model.tc_tslices, [10, 20, 30, 40, 41, 42])
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 0.35, 0.45, 0.45, 0.45])

    def test_apply_tc_replace_and_append_tslices(self):
        self.model.update_tc(tslices=[25, 35, 41, 42])
        self.assertEqual(self.model.tc_tslices, [10, 20, 25, 35, 41, 42])
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 0.35, 0.45, 0.45, 0.45])

    def test_apply_tc_truncate_tslices(self):
        self.model.update_tc(tslices=[20])
        self.assertEqual(self.model.tc_tslices, [10, 20])
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25])

    def test_apply_tc_truncate_and_append_tslices(self):
        self.model.update_tc(tslices=[15, 25, 41, 42, 43])
        self.assertEqual(self.model.tc_tslices, [10, 15, 25, 41, 42, 43])
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 0.35, 0.45, 0.45, 0.45])

    # both tc and tslices
    def test_apply_tc_replace_tc_and_tslices(self):
        new_tc = [1.05, 1.15, 1.25, 1.35, 1.45]
        new_tslices = [10, 120, 130, 140]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, new_tc)
        self.assertEqual(self.model.tc_tslices, new_tslices)

    def test_apply_tc_replace_tc_and_tslices_partial(self):
        new_tc = [1.05, 1.15, 1.25, 1.35, 1.45]
        new_tslices = [20, 130, 140]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, new_tc)
        self.assertEqual(self.model.tc_tslices, [10, 20, 130, 140])

    def test_apply_tc_replace_tc_partial_and_tslices(self):
        new_tc = [1.35, 1.45]
        new_tslices = [10, 120, 130, 140]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 1.35, 1.45])
        self.assertEqual(self.model.tc_tslices, new_tslices)

    def test_apply_tc_replace_tc_partial_and_tslices_partial_same(self):
        new_tc = [1.35, 1.45]
        new_tslices = [20, 130, 140]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 1.35, 1.45])
        self.assertEqual(self.model.tc_tslices, [10, 20, 130, 140])

    def test_apply_tc_replace_tc_partial_and_tslices_partial_tc_longer(self):
        new_tc = [1.25, 1.35, 1.45]
        new_tslices = [20, 130, 140]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 1.25, 1.35, 1.45])
        self.assertEqual(self.model.tc_tslices, [10, 20, 130, 140])

    def test_apply_tc_replace_tc_partial_and_tslices_partial_tslices_longer(self):
        new_tc = [1.45]
        new_tslices = [20, 130, 140]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 0.35, 1.45])
        self.assertEqual(self.model.tc_tslices, [10, 20, 130, 140])

    def test_apply_tc_replace_tc_append_slices(self):
        new_tc = [1.05, 1.15, 1.25, 1.35, 1.45, 1.46, 1.47]
        new_tslices = [41, 42]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, new_tc)
        self.assertEqual(self.model.tc_tslices, [10, 20, 30, 40, 41, 42])

    def test_apply_tc_replace_partial_small_tc_append_slices(self):
        new_tc = [1.47]
        new_tslices = [41, 42]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 0.35, 0.45, 0.45, 1.47])
        self.assertEqual(self.model.tc_tslices, [10, 20, 30, 40, 41, 42])

    def test_apply_tc_replace_partial_large_tc_append_slices(self):
        new_tc = [1.35, 1.45, 1.46, 1.47]
        new_tslices = [41, 42]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 1.35, 1.45, 1.46, 1.47])
        self.assertEqual(self.model.tc_tslices, [10, 20, 30, 40, 41, 42])

    def test_apply_tc_replace_tc_truncate_tslices(self):
        new_tc = [1.05, 1.12, 1.17, 1.27]
        new_tslices = [15, 25]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [1.05, 1.12, 1.17, 1.27])
        self.assertEqual(self.model.tc_tslices, [10, 15, 25])

    def test_apply_tc_replace_partial_small_tc_truncate_tslices(self):
        new_tc = [1.27]
        new_tslices = [15, 25]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 0.25, 1.27])
        self.assertEqual(self.model.tc_tslices, [10, 15, 25])

    def test_apply_tc_replace_partial_large_tc_truncate_tslices(self):
        new_tc = [1.17, 1.27]
        new_tslices = [15, 25]
        self.model.update_tc(tc=new_tc, tslices=new_tslices)
        self.assertEqual(self.model.tc, [0.05, 0.15, 1.17, 1.27])
        self.assertEqual(self.model.tc_tslices, [10, 15, 25])
