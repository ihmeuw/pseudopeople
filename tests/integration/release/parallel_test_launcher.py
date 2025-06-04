import csv
import glob
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd


def create_slurm_script(row: dict[str, str], script_name: str, output_dir: str) -> None:
    dataset = row["dataset"]
    pop = row['population']
    state = row['state'] 
    year = row['year']    
    engine = row["engine"]
    memory = row["memory"]
    time_limit = row["time"]  # DD:HH:MM
    partition = (
        "all.q" if int(time_limit.split(":")[0]) <= 24 else "long.q"
    )  # long.q if 24 longer than 24 hours
    release_tests_dir = Path(__file__).parent
    state_flag = f"--state {state}" if state != 'none' else ""
    year_flag = f"--year {year}" if year != 'default' else ""    
    pytest_command = (
        f"pytest -rA --release --dataset {dataset} --engine {engine} --population {pop} {state_flag} {year_flag} {release_tests_dir}"
    )
    # TODO: define cpus per task based on engine (ask Zeb)
    slurm_script = f"""#!/bin/bash 
#SBATCH --job-name=pytest_{dataset}_{engine} 
#SBATCH --output={output_dir}/pytest_{dataset}_{engine}.out 
#SBATCH --error={output_dir}/pytest_{dataset}_{engine}.err 
#SBATCH --mem={memory}G
#SBATCH --time={time_limit} 
#SBATCH --account=proj_simscience_prod
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=1
        
echo "Running pytest for dataset {dataset} with engine {engine}" 
{pytest_command} 
    """
    # Ensure the directory we want to write to exists
    script_path = f"{output_dir}/tmp_scripts/{script_name}"
    directory = os.path.dirname(script_path)
    os.makedirs(directory, exist_ok=True)

    # Write the script to a file
    with open(script_path, "w") as script:
        script.write(slurm_script)


def submit_slurm_job(script_name: str) -> str | None:
    try:
        result = subprocess.run(
            f"sbatch {script_name}", shell=True, check=True, capture_output=True
        )
        print(f"Submitted job: {script_name}")

        # Extract the job ID from the output
        output = result.stdout.strip()
        print(f"Submission successful: {output.decode('utf-8')}")

        # Slurm output is typically in the format "Submitted batch job <job_id>"
        job_id = output.split()[-1]  # Extract the last word (the job ID)
        return job_id.decode("utf-8")  # return bytes as string
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {script_name}\nError: {e}")
        return None


def is_job_running(job_id: str) -> bool:
    """
    Checks if a job with the given Slurm job ID is still running or pending.

    Args:
        job_id (str): Slurm job ID.

    Returns:
        bool: True if the job is running or pending; False if it has finished or if checking the job status fails.
    """
    try:
        result = subprocess.run(
            f"squeue --job {job_id}", shell=True, capture_output=True, text=True
        )
        return (
            job_id in result.stdout
        )  # The job ID is listed in squeue if it's running/pending
    except subprocess.CalledProcessError as e:
        print(f"Error checking job status: {e}")
        return False


def wait_for_all_jobs(job_ids: Sequence[str | None], poll_interval: int = 1800) -> None:
    """
    Waits for all jobs to complete, checking their status every `poll_interval` seconds.

    Args:
        job_ids (list): List of Slurm job IDs to monitor.
        poll_interval (int): Time interval (in seconds) between status checks. Default is 1800 seconds (30 minutes).
    """
    remaining_jobs = set(
        x for x in job_ids if x is not None
    )  # Track jobs that are still running
    while remaining_jobs:
        print(f"Checking status of {len(remaining_jobs)} remaining jobs...")

        completed_jobs = []
        for job_id in remaining_jobs:
            if not is_job_running(job_id):
                print(f"Job {job_id} has finished.")
                completed_jobs.append(job_id)

        # Remove completed jobs from the remaining jobs set
        remaining_jobs -= set(completed_jobs)

        if remaining_jobs:
            print(
                f"{len(remaining_jobs)} jobs have not finished. Checking again in {poll_interval // 60} minutes..."
            )
            time.sleep(poll_interval)  # Wait for the specified interval before checking again


def parse_outputs(output_dir: str, job_ids: list[str] | None = None) -> None:
    """Parses the output files to see how many jobs failed, didn't complete, and succeeded
    in the specified directory, prints this to screen, and writes a summary CSV with this info.

    Args:
        output_dir (str): Directory where Slurm output/error files are stored.
        job_ids (list[str] | None): List of Slurm job IDs to check. If provided, will verify that all expected outputs are present.
    """
    job_info = []

    output_files = glob.glob(f"{output_dir}/*.out")
    if job_ids:
        if len(output_files) < len(job_ids):
            print(
                f"Warning: Expected {len(job_ids)} job outputs, but found only {len(output_files)} in {output_dir}."
            )
            print(
                "This may indicate that some outputs are still being written. Rerun parse_outputs after all outputs have been written"
            )
            return
        else:
            print("All expected output files found.")

    # iterate through all files/jobs
    for file in output_files:
        with open(file, "r") as f:
            lines = f.readlines()
            if not lines:
                continue  # Skip empty files
            last_line = lines[-1].strip()

        if not (last_line.startswith("=") and last_line.endswith("=")):
            outcome = "not_completed"
        elif "failed" in last_line.lower():
            outcome = "failed"
        else:
            outcome = "passed"
        job_info.append({"outcome": outcome, "filepath": str(file)})

    job_info_df = pd.DataFrame(job_info).sort_values(by="outcome")
    job_info_df.to_csv(f"{output_dir}/summary_results.csv", index=False)

    num_failures = sum(job_info_df["outcome"] == "failed")
    num_incomplete = sum(job_info_df["outcome"] == "not_completed")
    if num_failures:
        print(
            f"FAIL: {num_failures} tests failed and {num_incomplete} tests did not finish. See {output_dir}/summary_results.csv for details."
        )
    else:
        print("All submitted tests passed successfully.")


if __name__ == "__main__":
    #csv_file = "data/parameters.csv
    csv_file = "data/test.csv"
    timestamp = datetime.now().strftime("%d-%H-%M-%S")
    output_dir = f"/mnt/team/simulation_science/priv/engineering/pseudopeople_release_testing/logs/{timestamp}"  # Directory where Slurm output/error files are stored
    job_ids = []
    submission_failures = []

    # Step 1: Read the CSV and generate and run Slurm scripts
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)  # Convert CSV rows into a list of dictionaries

        for i, row in enumerate(rows):
            script_name = f"job_{i}.sh"
            create_slurm_script(row, script_name, output_dir)  # Create the Slurm job script
            script_path = f"{output_dir}/tmp_scripts/{script_name}"
            job_id = submit_slurm_job(script_path)  # Submit the script
            if job_id:
                job_ids.append(job_id)  # Save the job
            else:
                submission_failures.append(row)

    # log submission failures
    print(f"{len(submission_failures)} submission failures: \n")
    for job_parameters in submission_failures:
        job_msg = ",".join(f"{key}={value}" for key, value in job_parameters.items())
        print(f"{job_msg} \n")

    # write submission failures to a CSV file
    if len(submission_failures) > 0:
        pd.DataFrame(submission_failures).to_csv(
            f"{output_dir}/submission_failures.csv", index=False
        )

    # Step 2: Wait for jobs to complete
    wait_for_all_jobs(job_ids, poll_interval=60)

    # Step 3: delete tmp_scripts directory
    scripts_dir = f"{output_dir}/tmp_scripts"
    try:
        shutil.rmtree(scripts_dir)
    except:
        print(f"Failed to delete temporary scripts directory: {scripts_dir}")

    # Step 4: Parse outputs and write summary CSV
    parse_outputs(output_dir, job_ids)
