import unittest
import json
import os


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # load credentials and set environment variables
        with open('../creds.json') as creds_file:
            for cred_key, cred_val in json.load(creds_file).items():
                os.environ[cred_key] = cred_val

    def region_test(self, region=None, batch_size=6, increment_size=2, write_batch_output=False):
        os.chdir('../covid_model')
        run_str = f"python run_fit.py -om --fit_id 553 --batch_size {batch_size}"\
                  f" --increment_size {increment_size}" \
                  f" --params input/params.json " \
                  f" --refresh_vacc "
        if region is not None:
            run_str += f"--region {region} --region_params input/region_params.json"
            run_str += f" --hosp_data ../../data/processed_lpha_hospbycounty_20220128.csv"
        if write_batch_output:
            run_str += " --write_batch_output"
        output = os.system(run_str)
        if output == 1:
            # try again (could be db connection issue)
            output = os.system(run_str)
        if output == 1:
            self.fail(f"Failed: {region}")

    def test_ad(self):
        self.region_test("ad")

    def test_ar(self):
        self.region_test("ar")

    def test_bo(self):
        self.region_test("bo")

    def test_br(self):
        self.region_test("br")

    def test_den(self):
        self.region_test("den")

    def test_doug(self):
        self.region_test("doug")

    def test_ep(self):
        self.region_test("ep")

    def test_jeff(self):
        self.region_test("jeff")

    def test_lar(self):
        self.region_test("lar")

    def test_mesa(self):
        self.region_test("mesa")

    def test_pueb(self):
        self.region_test("pueb")

    def test_weld(self):
        self.region_test("weld")

    def test_cent(self):
        self.region_test("cent")

    def test_cm(self):
        self.region_test("cm")

    def test_met(self):
        self.region_test("met")

    def test_ms(self):
        self.region_test("ms")

    def test_ne(self):
        self.region_test("ne")

    def test_nw(self):
        self.region_test("nw")

    def test_slv(self):
        self.region_test("slv")

    def test_sc(self):
        self.region_test("sc")

    def test_sec(self):
        self.region_test("sec")

    def test_sw(self):
        self.region_test("sw")

    def test_wcp(self):
        self.region_test("wcp")

    def test_co(self):
        self.region_test()


if __name__ == '__main__':
    unittest.main()
