#=======================================================================================================================
# docker_wrapper_test.py
# Written by: Andrew Hill
# Last Modified: 1/23/2022
# Description:
#   This file allows us to test running docker_wrapper.py without needing to use a Docker container or pass in a Base64
#   encoded string. This file simply reads a JSON parameter file, and runs the wrapper from docker_wrapper.py using
#   the arguments specified in the file.
#=======================================================================================================================
import json
from docker_wrapper import wrapper_run

if __name__ == "__main__":
    # Load a parameter JSON file.
    with open("sample_config.json","r") as f:
        args = json.load(f)
    # Run the wrapper with this input file.
    wrapper_run(args)
