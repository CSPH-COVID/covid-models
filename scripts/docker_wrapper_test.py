#=======================================================================================================================
# docker_wrapper_test.py
# Written by: Andrew Hill
# Last Modified: 1/23/2022
# Description:
#   This file allows us to test running docker_wrapper.py without needing to use a Docker container or pass in a Base64
#   encoded string. This file simply reads a JSON parameter file, and runs the wrapper from docker_wrapper.py using
#   the arguments specified in the file.
#=======================================================================================================================
import os
import json
from docker_wrapper import wrapper_run
from google.cloud import bigquery

if __name__ == "__main__":
    # Get max spec_id
    bq_client = bigquery.Client()
    first_new_id = list(bq_client.query("SELECT MAX(spec_id)+1 FROM co-covid-models.covid_model.specifications").result())[0][0]
    os.environ["SPEC_ID"] = str(int(first_new_id))
    print(f"Using spec_id {first_new_id} as base model spec_id")
    #os.environ["SPEC_ID"] = "4692"
    # Load a parameter JSON file.
    with open("sample_config.json","r") as f:
        args = json.load(f)
    # Run the wrapper with this input file.
    wrapper_run(args)
