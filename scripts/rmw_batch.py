# =======================================================================================================================
# rmw_batch.py
# Written by: Andrew Hill
# Last Modified: 1/20/2023
# Description:
#   This script will start a Google Batch job to run a set of RMW models in parallel, monitor their execution, and
#   collect the results when they finish executing.
# =======================================================================================================================
import time
import json
import base64
import argparse
import subprocess
from typing import List, Union, Optional
from os.path import join
from datetime import datetime, timezone
from collect_files import combine_files
from google.cloud.storage import Blob, Client as GCS_Client
from google.cloud import batch_v1 as batch, bigquery # If you get error here, try 'pip install google-cloud-batch'

# Default parameters which are used if arguments are not passed to the script.
PROJECT_ID = "co-covid-models"
GCP_REGION = "us-central1"
DEFAULT_JOB_ID = "rmw-model-run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
BUCKET_NAME = "covid-rmw-model-results"
# Default parameters, these should not be changed.
BUCKET_MOUNT_PATH = "/covid_rmw_model/covid_model/output"
DOCKER_BUILD_SCRIPT = "docker_build_and_deploy.sh"
LINE_UP = "\033[F"
LINE_CLEAR = "\033[K"
LINE_UP_CLEAR = LINE_CLEAR + LINE_UP
RED = "\033[31m"
GREEN = "\033[32m"
OFF = "\033[0m"


def make_status_table(tasks, regions) -> List[str]:
    headers = ["   ", "TASK ID", "REGION", "STATE"]
    progress = ["|", "/", "—", "\\"]
    col = " | "
    lbar = "| "
    rbar = " |"
    # Sort by task index.
    tasks = sorted(tasks,key=lambda elem: int(elem.name.rsplit("/",maxsplit=1)[1]))
    task_ids = [t.name.rsplit("/", maxsplit=1)[-1] for t in tasks]
    tid_col_width = max(max(len(tid) for tid in task_ids), len(headers[1]))

    reg_col_width = max(max(len(reg) for reg in regions), len(headers[2]))

    all_states = tasks[0].status.State
    status_col_width = max(max(len(str(x)) for x in all_states), len(headers[3]))
    task_status = [str(task.status.state) for task in tasks]

    widths = [3, tid_col_width, reg_col_width, status_col_width]

    pad_header = [header.center(width) for header, width in zip(headers, widths)]
    header_bar = lbar + col.join(pad_header) + rbar

    rows = []
    for tid, region, status in zip(task_ids, regions, task_status):
        p = (progress[int(time.perf_counter() * 2) % 4] if status == "State.RUNNING" else " ").center(3)
        tid_c = tid.ljust(tid_col_width)
        region_c = region.ljust(reg_col_width)

        status_c = status.center(status_col_width)
        if status in ["State.FAILED", "State.STATE_UNSPECIFIED"]:
            status_c = RED + status_c + OFF
        elif status in ["State.SUCCEEDED", "State.RUNNING"]:
            status_c = GREEN + status_c + OFF
        rows.append(lbar + col.join([p, tid_c, region_c, status_c]) + rbar)

    status_table_str = [header_bar] + rows

    return status_table_str


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


def decode_arguments(job: batch.Job) -> dict:
    """ Extract the JSON parameters from an existing Job instance
    :param job: The Job instance to extract from
    :return: A dictionary containing the job parameters.
    """
    job_command = job.task_groups[0].task_spec.runnables[0].container.commands[0]
    job_command_str = base64.b64decode(job_command)
    job_command_dict = json.loads(job_command_str)
    return job_command_dict


def get_new_spec_id() -> int:
    """ Acquire an unused spec_id from BigQuery, which will become the base for all spec_ids generated in this run.

    :return: An unused spec_id
    """
    bq_client = bigquery.Client()
    query_job = bq_client.query(query="SELECT MAX(spec_id)+1 FROM `co-covid-models.covid_model.specifications`")
    query_result = list(query_job.result())[0][0]
    return query_result


def monitor_batch_job(batch_job: batch.Job, params: dict, update_every: Union[int, float] = 10, print_every: Union[int, float] = 0.5):
    """ Monitors the state of a Batch Job passed in as an argument, and logs the status of each Task to the console.
    :param batch_job: The Batch Job intance to monitor.
    :param params: Dictionary of parameters for the job to monitor.
    :param update_every: How often to query Google Cloud for an updated Job status. This is calculated relative to the
                         Job update_time paramter. Defaults to every 10 seconds.
    :param print_every: How often to print to the screen. Defaults to 0.5s (twice per second).
    :return: None, when batch_job is finished.
    """
    # Wait for job to move out of QUEUED state
    stime = time.perf_counter()
    tasks_status = list(batch_client.list_tasks(parent=batch_job.name + "/taskGroups/group0"))
    while len(tasks_status) == 0:
        # Get updated job status.
        ctime = int(time.perf_counter() - stime)
        print(f"Waiting until Job has tasks listed... [{ctime:03d}s]", end="\r")
        time.sleep(5)
        tasks_status = list(batch_client.list_tasks(parent=batch_job.name + "/taskGroups/group0"))
    # Display the output status for each task in the job.
    last_update = batch_job.update_time
    status_output = make_status_table(tasks=tasks_status, regions=params["regions"])
    print("\n".join(status_output))
    # While job is running
    while batch_job.status.state not in [batch_job.status.State.SUCCEEDED, batch_job.status.State.FAILED]:
        # If updated time is greater than the threshold, get updated job status from Batch API.
        if (datetime.now(timezone.utc) - last_update).seconds > update_every:
            batch_job = batch_client.get_job(name=batch_job.name)
            last_update = datetime.now(timezone.utc)
            tasks_status = list(batch_client.list_tasks(parent=batch_job.name + "/taskGroups/group0"))
        # Format the table that shows the status of all jobs.
        status_output = make_status_table(tasks=tasks_status, regions=params["regions"])
        # Clear the screen and print the new output.
        print(LINE_UP_CLEAR * (len(status_output) + 1), end="")
        print("\n".join(status_output))
        print(f"Last Updated: {last_update}")
        # Sleep until the next print cycle.
        time.sleep(print_every)


def batch_run_rmw_model(spec_id: int,
                        params: str,
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
                        task_count_per_node: Optional[int] = None,
                        parallelism: Optional[int] = None,
                        client: batch.BatchServiceClient = None):
    """
    :param spec_id: The spec_id which will be used to generate sequential increasing spec_ids for all parallel runs.
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
    :param client: The BatchServiceClient which will be used to run this job.
    :return:
    """

    # Create Batch Client
    client = client if client is not None else batch.BatchServiceClient()

    # Create Runnable
    runnable = batch.Runnable()
    runnable.container = batch.Runnable.Container()
    runnable.container.image_uri = "gcr.io/co-covid-models/github.com/csph-covid/covid-rmw-model:latest"
    runnable.container.commands = [params]
    runnable.container.volumes = [f"/mnt/disks/output:{bucket_mount_path}:rw"]

    # Create Task Spec
    task = batch.TaskSpec()
    task.runnables = [runnable]
    task.environment.variables["SPEC_ID"] = str(spec_id)

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
    if task_count_per_node is not None:
        group.task_count_per_node = task_count_per_node
    if parallelism is not None:
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

    return client.create_job(request=request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command",
                                       required=True)

    # Job Creation
    create_job_parser = subparsers.add_parser("create",
                                              help="create help")
    create_job_parser.add_argument("--build_and_deploy",
                                   action="store_true",
                                   help=f"If set, run the '{DOCKER_BUILD_SCRIPT}' script before running to ensure that"
                                        " the most recent code is used for the image.")
    create_job_parser.add_argument("--job_params",
                                   type=str,
                                   help="Name of a JSON file containing job parameters to read in.",
                                   default="sample_config.json")
    create_job_parser.add_argument("--project_id",
                                   type=str,
                                   help="The project id for the Google Cloud Project",
                                   default=PROJECT_ID)
    create_job_parser.add_argument("--region",
                                   type=str,
                                   help="The Google Cloud region to run the job under.",
                                   default=GCP_REGION)
    create_job_parser.add_argument("--bucket_name",
                                   type=str,
                                   help="The name of the Google Cloud Storage bucket where the results will be"
                                        " uploaded.",
                                   default=BUCKET_NAME)
    create_job_parser.add_argument("--job_id",
                                   type=str,
                                   help="Defines the id of the Batch job, and the name of the output "
                                        "directory under BUCKET_NAME. ",
                                   default=DEFAULT_JOB_ID)

    # Job Monitoring
    monitor_job_parser = subparsers.add_parser("monitor",
                                               help="monitor help")
    monitor_job_parser.add_argument("--job_id",
                                    type=str,
                                    help="The Job ID of the job we should monitor.",
                                    required=True)

    args = parser.parse_args()

    # Create batch client and run batch job
    batch_client = batch.BatchServiceClient()
    if args.command == "create":
        # Create and submit a new Batch Job.
        if args.build_and_deploy:
            print(f"--build_and_deploy flag is set, running '{DOCKER_BUILD_SCRIPT}'...")
            subprocess.run(join(".", DOCKER_BUILD_SCRIPT), shell=True, check=True)
            print("Build script completed.")

        # Acquire the spec_id from BigQuery
        new_spec_id = get_new_spec_id()
        # Encode job parameters.
        # TODO: Make this a little less silly. We parse the JSON here, then convert it back to string inside
        #       the encode_arguments function.
        with open(args.job_params, "r") as f:
            job_params = json.load(f)
        enc_job_params = encode_arguments(obj=job_params)

        # Create output path in GCS. Path will be BUCKET/JOB_ID
        bucket_output_path = create_bucket_subdir(bucket=args.bucket_name, subdir=args.job_id)

        # job = batch_client.get_job(name="projects/co-covid-models/locations/us-central1/jobs/rmw-model-run-20230110-114102")
        job = batch_run_rmw_model(spec_id=new_spec_id,
                                  params=enc_job_params,
                                  project_id=args.project_id,
                                  region=args.region,
                                  bucket_path=bucket_output_path,
                                  task_count=len(job_params["regions"]),
                                  job_id=args.job_id,
                                  bucket_mount_path=BUCKET_MOUNT_PATH,
                                  machine_type="e2-highcpu-4",
                                  cpu_milli=4000,
                                  memory_mib=4096,
                                  max_run_duration="18000s",
                                  client=batch_client)
    else:
        # Monitor an existing Batch Job.
        job_name_str = f"projects/{PROJECT_ID}/locations/{GCP_REGION}/jobs/{args.job_id}"
        job = batch_client.get_job(name=job_name_str)
        job_params = decode_arguments(job=job)

    # Log the table status while the job is running.
    monitor_batch_job(batch_job=job, params=job_params)
    print("Batch job complete, collecting files...")
    # Combine files when job is complete
    #combine_files(project=args.project_id, bucket=args.bucket_name, subdir=args.job_id)

    # TODO: Create BigQuery tables for outputs based on the combined files from google cloud storage.

    print("Done!")
