import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "pytest_args",
    [
        ([]),
        (["--dataset", "acs"]),
        (["--dataset", "cps"]),
        # (["--dataset", "acs", "--population", "USA"]),
        # (["--dataset", "acs", "--population", "USA", "--state", "RI"]),
        (["--dataset", "wic", "--year", "2015"]),
        # (["--dataset", "wic", "--population", "USA", "--state", "RI", "--year", "2015"]),
    ],
    ids=['1','2','3','4']
)
def test_release_tests(pytest_args: list[str], release_logging_dir: Path, request) -> None:
    os.chdir(Path(__file__).parent)  # need this to access cli options from conftest.py
    base_cmd = ["pytest", "--release", "test_release.py"]
    cmd = base_cmd + pytest_args
    job_id = request.node.callspec.id
    log_file = f"{release_logging_dir}/pytest_{job_id}.o"
    # Open a file in write mode
    with open(log_file, 'w') as file:
        # Run pytest and direct stdout to the log file
        subprocess.run(cmd, stdout=file)


@pytest.mark.parametrize("dataset", ["acs", "cps"])
def test_slow_tests(dataset: str) -> None:
    os.chdir(Path(__file__).parent)  # need this to access cli options from conftest.py
    cmd = ["pytest", "--runslow", "test_release.py", "--dataset", dataset]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
