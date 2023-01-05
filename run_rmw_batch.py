# =======================================================================================================================
# run_rmw_batch.py
# Written by: Andrew Hill
# Last Modified: 1/4/2023
# Description:
#   This script will start a Google Batch job to run a set of RMW models in parallel, monitor their execution, and
#   collect the results when they finish executing.
# =======================================================================================================================
import json
import base64
import argparse
import subprocess
import curses  # If you are running on Windows curses is not included by default. Try 'pip install windows-curses'.
from os.path import join
from datetime import datetime
from collect_files import combine_files
from google.cloud.storage import Blob, Client as GCS_Client
from google.cloud import batch_v1 as batch  # If you get error here, try 'pip install google-cloud-batch'

# Default parameters which are used if arguments are not passed to the script.
PROJECT_ID = "co-covid-models"
GCP_REGION = "us-central1"
DEFAULT_JOB_ID = "rmw-model-run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
BUCKET_NAME = "covid-rmw-model-results"
# Default parameters, these should not be changed.
BUCKET_MOUNT_PATH = "/covid_rmw_model/covid_model/output"
DOCKER_BUILD_SCRIPT = "docker_build_and_deploy.sh"


def create_bucket_subdir(bucket: str, subdir: str):
    """
    :param bucket: Name of the bucket to create the subdirectory within.
    :param subdir: Name of the subdirectory to be created.
    :return: The path to the subdirectory.
    """
    client = GCS_Client(project=PROJECT_ID)

    bucket_obj = client.get_bucket(bucket_or_name=bucket)

    subdir_name = subdir + "/"
    subdir_blob = Blob(name=subdir_name, bucket=bucket_obj)
    subdir_blob.upload_from_string("", content_type="media")

    return bucket + "/" + subdir_name


def encode_arguments(obj: dict):
    """
    :param obj: Dictionary of JSON-serializable data which will be encoded to Base64.
    :return: Base64-encoded data.
    """
    json_enc = json.dumps(obj).encode("utf-8")
    b64_enc = base64.b64encode(json_enc)
    return b64_enc.decode("utf-8")


def batch_run_rmw_model(params: str,
                        project_id: str,
                        region: str,
                        bucket_path: str,
                        task_count: int,
                        job_id: str,
                        bucket_mount_path: str,
                        machine_type: str,
                        cpu_milli: int,
                        memory_mib: int,
                        max_run_duration: str,
                        task_count_per_node: int,
                        parallelism: int):
    """
    :param params: The dictionary of parameters to pass to each Batch job.
    :param project_id: The Google Cloud Project ID for the job.
    :param region: The Google Cloud region to run the job under.
    :param bucket_path: Path to a Google Cloud Storage bucket where the results should be written.
    :param task_count: Number of tasks which should be run. This should match the number of regions in args["region"]
    :param job_id: Unique id/name for this job, will also be the name of the subdirectory that contains the output under
                   bucket_path.
    :param bucket_mount_path: The path within the container where the model's output files are written.
    :param machine_type: Defines which GCP machine types will be used for the job.
    :param cpu_milli: Milli-cpu limit for each task.
    :param memory_mib: Memory (MiB) limit for each task.
    :param max_run_duration: Maximum duration that a single task can run before it is killed.
    :param task_count_per_node: Number of tasks which should run simultaneously on an instance.
    :param parallelism: Maximum number of instances which will be run simultaneously.
    :return:
    """

    # Create Batch Client
    client = batch.BatchServiceClient()

    # Create Runnable
    runnable = batch.Runnable()
    runnable.container = batch.Runnable.Container()
    runnable.container.image_uri = "us-central1-docker.pkg.dev/co-covid-models/cste/rmw_model:latest"
    runnable.container.commands = [params]
    runnable.container.volumes = [f"/mnt/disks/output:{bucket_mount_path}:rw"]

    # Create Task Spec
    task = batch.TaskSpec()
    task.runnables = [runnable]

    # Set up compute resources
    resources = batch.ComputeResource()
    resources.cpu_milli = cpu_milli
    resources.memory_mib = memory_mib
    task.compute_resource = resources
    task.max_run_duration = max_run_duration

    # Set up the mapped GCS volume so that results go to Google Cloud Storage.
    volume = batch.Volume()
    gcs = batch.GCS()
    gcs.remote_path = bucket_path
    volume.mount_path = "/mnt/disks/output"
    volume.gcs = gcs
    task.volumes = [volume]

    # Set up TaskGroup
    group = batch.TaskGroup()
    group.task_spec = task
    group.task_count = task_count
    group.task_count_per_node = task_count_per_node
    group.parallelism = parallelism

    # Set up AllocationPolicy
    policy = batch.AllocationPolicy.InstancePolicy()
    policy.machine_type = machine_type
    instances = batch.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    alloc_policy = batch.AllocationPolicy()
    alloc_policy.instances = [instances]

    # Create Job
    job = batch.Job()
    job.task_groups = [group]
    job.allocation_policy = alloc_policy
    job.logs_policy = batch.LogsPolicy()
    job.logs_policy.destination = batch.LogsPolicy.Destination.CLOUD_LOGGING

    # Create Job Request
    request = batch.CreateJobRequest()
    request.job = job
    request.job_id = job_id
    request.parent = f"projects/{project_id}/locations/{region}"

    return client.create_job(request=request),


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_and_deploy",
                        action="store_true",
                        help=f"If set, run the '{DOCKER_BUILD_SCRIPT}' script before running to ensure that the most"
                             " recent code is used for the image.")
    parser.add_argument("--job_params",
                        type=str,
                        help="Name of a JSON file containing job parameters to read in.",
                        default="sample_config.json")
    parser.add_argument("--regions",
                        type=str,
                        nargs="+",
                        help="Which regions to run the model for.",
                        default=[])
    parser.add_argument("--project_id",
                        type=str,
                        help="The project id for the Google Cloud Project",
                        default=PROJECT_ID)
    parser.add_argument("--region",
                        type=str,
                        help="The Google Cloud region to run the job under.",
                        default=GCP_REGION)
    parser.add_argument("--bucket_name",
                        type=str,
                        help="The name of the Google Cloud Storage bucket where the results will be uploaded.",
                        default=BUCKET_NAME)
    parser.add_argument("--job_id",
                        type=str,
                        help="Defines the id of the Batch job, and the name of the output "
                             "directory under BUCKET_NAME. ",
                        default=DEFAULT_JOB_ID)
    args = parser.parse_args()

    if args.build_and_deploy:
        print(f"--build_and_deploy flag is set, running '{DOCKER_BUILD_SCRIPT}'...")
        subprocess.run(join(".", DOCKER_BUILD_SCRIPT), shell=True, check=True)
        print("Build script completed.")

    # Encode job parameters.
    # TODO: Make this a little less silly. We parse the JSON here, then convert it back to string inside
    #       the encode_arguments function.
    with open(args.job_params,"r") as f:
        job_params = json.load(f)
    enc_job_params = encode_arguments(obj=job_params)

    # Create output path in GCS. Path will be BUCKET/JOB_ID
    bucket_output_path = create_bucket_subdir(bucket=args.bucket_name, subdir=args.job_id)

    # Run Batch Job
    request = batch_run_rmw_model(params=enc_job_params,
                                  project_id=args.project_id,
                                  region=args.region,
                                  bucket_path=bucket_output_path,
                                  task_count=len(job_params["regions"]),
                                  job_id=args.job_id,
                                  bucket_mount_path=BUCKET_MOUNT_PATH,
                                  machine_type="e2-standard-4",
                                  cpu_milli=4000,
                                  memory_mib=16384,
                                  max_run_duration="18000s",
                                  task_count_per_node=1,
                                  parallelism=6)
    # Display the output status for each task in the job.


    # Combine files when job is complete
    combine_files(project=args.project_id,bucket=args.bucket_name,subdir=args.job_id)
