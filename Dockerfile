# CSTE Rocky Mountain West Regional COVID Model Docker Image
# Written by: Andrew Hill
# Last Modified: 2023-01-20
# Description: This image wraps the CSTE RMW Regional Model, and can be used to run multiple instances of the model
#              in parallel (for example using Google Workflows and Google Batch).

# Build off of the Python 3.10 / Debian Bullseye image.
FROM python:3.10-bullseye
# Set a working directory.
WORKDIR /covid_rmw_model
# Copy the files over to the image.
# Note: We use a .dockerignore file to only copy relevant files from the covid_model directory.
COPY covid_model ./covid_model/
# Make a empty output directory. We need this so we can mount the GCS bucket here.
RUN mkdir covid_model/output
# Copy the wrapper and requirements file to the image.
COPY scripts/docker_wrapper.py scripts/docker_wrapper_test.py docker_test/sample_config.json requirements.txt ./
# Install required packages
RUN pip install --no-cache-dir -r requirements.txt
# Set the entrypoint for the image.
ENTRYPOINT ["python3","docker_wrapper.py"]
