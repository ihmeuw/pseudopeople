import os
import subprocess

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
    DEFAULT_ENGINE,
    DEFAULT_STATE,
    DEFAULT_YEAR,
    FULL_USA_FILEPATH,
    RI_FILEPATH,
    _parse_dataset_params,
)


@pytest.fixture()
def check_subprocess_environment():
    if "RUNNING_AS_SUBPROCESS" not in os.environ:
        pytest.skip("Skipping this test because it's not running as a subprocess")


EXPECTED_PARAMETERS = {
    "census": (
        generate_decennial_census,
        None,
        DEFAULT_ENGINE,
        DEFAULT_STATE,
        DEFAULT_YEAR,
    ),
    "acs": (
        generate_american_community_survey,
        FULL_USA_FILEPATH,
        DEFAULT_ENGINE,
        DEFAULT_STATE,
        DEFAULT_YEAR,
    ),
    "ssa": (generate_social_security, None, "dask", DEFAULT_STATE, DEFAULT_YEAR),
    "cps": (generate_current_population_survey, None, DEFAULT_ENGINE, "RI", DEFAULT_YEAR),
    "tax_w2_1099": (
        generate_taxes_w2_and_1099,
        None,
        DEFAULT_ENGINE,
        DEFAULT_STATE,
        2010,
    ),
    "wic": (generate_women_infants_and_children, None, DEFAULT_ENGINE, "RI", 2010),
    "tax_1040": (
        generate_taxes_1040,
        RI_FILEPATH,
        DEFAULT_ENGINE,
        DEFAULT_STATE,
        DEFAULT_YEAR,
    ),
}


@pytest.mark.subprocess_test
@pytest.mark.usefixtures("check_subprocess_environment")
def test_parsing_fixture_params(request) -> None:
    output = _parse_dataset_params(request)
    dataset_name = output[0]
    assert output[1:] == EXPECTED_PARAMETERS[dataset_name]


@pytest.mark.parametrize(
    "pytest_args",
    [
        (["--dataset", "census"]),
        (["--dataset", "acs", "--population", "USA"]),
        (["--dataset", "tax_1040", "--population", "RI"]),
        (["--dataset", "ssa", "--engine", "dask"]),
        (["--dataset", "cps", "--state", "RI"]),
        (["--dataset", "tax_w2_1099", "--year", "2010"]),
        (["--dataset", "wic", "--state", "RI", "--year", "2010"]),
    ],
)
def test_parsing_fixture_param_combinations(pytest_args) -> None:
    env = os.environ.copy()
    env["RUNNING_AS_SUBPROCESS"] = "1"
    base_cmd = ["pytest", "-k", "test_parsing_fixture_params"]
    cmd = base_cmd + pytest_args
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
