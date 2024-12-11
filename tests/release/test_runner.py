import os
import subprocess
import pytest
from pathlib import Path

@pytest.mark.parametrize(
    "pytest_args",
    [
        ([]),
        (["--dataset", "acs"]),
        (["--dataset", "cps"]),
        #(["--dataset", "acs", "--population", "USA"]),
        #(["--dataset", "acs", "--population", "USA", "--state", "RI"]),
        (["--dataset", "wic", "--year", "2015"]),
        #(["--dataset", "wic", "--population", "USA", "--state", "RI", "--year", "2015"]),
    ],
)
def test_integration(pytest_args: list[str]) -> None:
    os.chdir(Path(__file__).parent)  # need this to access options from conftest.py
    base_cmd = ["pytest", "--release", "test_release.py"]
    cmd = base_cmd + pytest_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
