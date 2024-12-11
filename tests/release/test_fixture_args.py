import os
import subprocess
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from tests.release.conftest import (
    CLI_DEFAULT_ENGINE,
    CLI_DEFAULT_STATE,
    CLI_DEFAULT_YEAR,
    FULL_USA_FILEPATH,
)


@pytest.fixture()
def check_subprocess_environment() -> None:
    if "RUNNING_AS_SUBPROCESS" not in os.environ:
        pytest.skip("Skipping this test because it's not running as a subprocess")


# expected parameters tuples contain
# (generating function, data source, year, state, engine)
EXPECTED_PARAMETERS = {
    "census": (
        generate_decennial_census,
        None,
        CLI_DEFAULT_YEAR,
        CLI_DEFAULT_STATE,
        CLI_DEFAULT_ENGINE,
    ),
    "acs": (
        generate_american_community_survey,
        FULL_USA_FILEPATH,
        CLI_DEFAULT_YEAR,
        CLI_DEFAULT_STATE,
        CLI_DEFAULT_ENGINE,
    ),
    "cps": (
        generate_current_population_survey,
        None,
        CLI_DEFAULT_YEAR,
        "RI",
        CLI_DEFAULT_ENGINE,
    ),
    "ssa": (generate_social_security, None, CLI_DEFAULT_YEAR, CLI_DEFAULT_STATE, "dask"),
    "tax_1040": (
        generate_taxes_1040,
        None,
        2010,
        CLI_DEFAULT_STATE,
        CLI_DEFAULT_ENGINE,
    ),
    "tax_w2_1099": (
        generate_taxes_w2_and_1099,
        None,
        2010,
        "RI",
        CLI_DEFAULT_ENGINE,
    ),
    "wic": (generate_women_infants_and_children, FULL_USA_FILEPATH, 2015, "MO", "dask"),
}


@pytest.mark.usefixtures("check_subprocess_environment")
def test_parsing_fixture_params(
    dataset_params: tuple[str | int | Callable[..., pd.DataFrame] | None, ...],
    request: pytest.FixtureRequest,
) -> None:
    # we know output will have a string as the first element but can't type this
    # while specifying the types of the other elements in output
    dataset_name: str = dataset_params[0]  # type: ignore [assignment]
    assert dataset_params[1:] == EXPECTED_PARAMETERS[dataset_name]


@pytest.mark.parametrize(
    "pytest_args",
    [
        (["--dataset", "census"]),
        (["--dataset", "acs", "--population", "USA"]),
        (["--dataset", "cps", "--state", "RI"]),
        (["--dataset", "ssa", "--engine", "dask"]),
        (["--dataset", "tax_1040", "--year", "2010"]),
        (["--dataset", "tax_w2_1099", "--state", "RI", "--year", "2010"]),
        (
            [
                "--dataset",
                "wic",
                "--population",
                "USA",
                "--engine",
                "dask",
                "--state",
                "MO",
                "--year",
                "2015",
            ]
        ),
    ],
)
def test_parsing_fixture_param_combinations(pytest_args: list[str]) -> None:
    env = os.environ.copy()
    env["RUNNING_AS_SUBPROCESS"] = "1"
    os.chdir(Path(__file__).parent)  # need this to access options from conftest.py
    base_cmd = ["pytest", "-k", "test_parsing_fixture_params"]
    cmd = base_cmd + pytest_args
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    del env["RUNNING_AS_SUBPROCESS"]
