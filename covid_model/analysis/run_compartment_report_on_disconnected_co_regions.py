import json
import os

from covid_model import all_regions

spec_ids = [1238, 1264, 1290, 1316, 1342, 1368, 1394, 1420, 1446, 1472, 1498]


def main():
    # load credentials and set environment variables
    print(os.getcwd())
    with open('../../creds.json') as creds_file:
        for cred_key, cred_val in json.load(creds_file).items():
            os.environ[cred_key] = cred_val
    os.chdir('../../covid_model')

    for spec_id, region in zip(spec_ids, all_regions):
        print(f' processing {region}')
        run_str = f"python analysis/run_compartment_report.py " \
                  f"--from_specs {spec_id} " \
                  f"--model_class RegionalCovidModel " \
                  f"--region {region} " \
                  f"--save_prefix {region} "
        output = os.system(run_str)


if __name__ == '__main__':
    main()
