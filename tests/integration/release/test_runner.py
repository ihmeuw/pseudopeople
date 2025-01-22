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
)
def test_release_tests(pytest_args: list[str], release_output_dir: Path) -> None:
    os.chdir(Path(__file__).parent)  # need this to access cli options from conftest.py
    base_cmd = ["pytest", "--release", "test_release.py"]
    cmd = base_cmd + pytest_args
    # Open a file in write mode
    with open(f"{release_output_dir}/pytest.new", 'w') as log_file:
        # Run pytest and direct stdout to the log file
        subprocess.run(cmd, stdout=log_file)


@pytest.mark.parametrize("dataset", ["acs", "cps"])
def test_slow_tests(dataset: str) -> None:
    os.chdir(Path(__file__).parent)  # need this to access cli options from conftest.py
    cmd = ["pytest", "--runslow", "test_release.py", "--dataset", dataset]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
