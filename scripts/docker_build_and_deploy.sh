#!/bin/bash
# This script builds a Docker image from this repository and deploys it to Google Artifact Registry.
# Make sure that you have:
# 1. Installed Docker on your local machine.
# 2. Installed the gcloud CLI on your local machine.
# 3. Run `gcloud auth configure-docker us-central1-docker.pkg.dev`
# before running this script.

IMAGE_TAG="rmw_model"
GCP_REGION="us-central1"
GCP_PROJECT="co-covid-models"
GCP_ARTIFACT_REPO="cste"
GCP_ARTIFACT_REGISTRY_PATH="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_ARTIFACT_REPO}/${IMAGE_TAG}"
# Begin the script
echo "======================Run Configuration======================"
echo "Docker Local Image Tag:  ${IMAGE_TAG}"
echo "GCP Region: ${GCP_REGION}"
echo "GCP Project Name: ${GCP_PROJECT}"
echo "Artifact Registry Target URL: ${GCP_ARTIFACT_REGISTRY_PATH}"
echo "============================================================="
# Build the Docker image.
echo "Building Docker image..."
docker buildx build --platform=linux/amd64 -t ${IMAGE_TAG} .
echo "Build complete."
# Tag image with remote repo name and push image.
echo "Tagging '${IMAGE_TAG}' with the Artifact Registry path..."
docker tag ${IMAGE_TAG} ${GCP_ARTIFACT_REGISTRY_PATH}
echo "Pushing image to Artifact Registry at location: ${GCP_ARTIFACT_REGISTRY_PATH}."
echo "If the script fails at this step, please make sure you have configured"
echo "Docker and gcloud CLI as described in the comment at the top of this script."
docker push ${GCP_ARTIFACT_REGISTRY_PATH}
echo "Script complete."