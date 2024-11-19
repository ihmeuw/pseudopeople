import functools
import os
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from memory_profiler import memory_usage

from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)


def pytest_addoption(parser):
    parser.addoption(
        "--year",
        action="store",
        default=2020,
        help="The year to subset our data to.",
    )
    parser.addoption(
        "--state",
        action="store",
        default=None,
        help="The dataset to generate.",
    )
    parser.addoption(
        "--dataset",
        action="store",
        help="The dataset to generate.",
    )

############
# Fixtures #
############
@pytest.fixture(scope="session")
def output_dir() -> Path:
    #output_dir = os.environ.get("PSP_TEST_OUTPUT_DIR")
    output_dir = '/home/hjafari/ppl_testing'
    if not output_dir:
        raise ValueError("PSP_TEST_OUTPUT_DIR environment variable not set")
    output_dir = Path(output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


@pytest.fixture(scope="session")
def american_community_survey(output_dir, request):
    year = int(request.config.getoption("--year", default=2020))
    state = request.config.getoption("--state", default=None)
    FULL_USA_FILEPATH = '/mnt/team/simulation_science/pub/models/vivarium_census_prl_synth_pop/results/release_02_yellow/full_data/united_states_of_america/2023_08_21_16_35_27/final_results/2023_08_31_15_58_01/pseudopeople_simulated_population_usa_2_0_0'
    return profile_data_generation(output_dir)(generate_american_community_survey)(source=FULL_USA_FILEPATH, year=year, state=state)


@pytest.fixture(scope="session")
def current_population_survey(output_dir, request):
    year = int(request.config.getoption("--year", default=2020))
    state = request.config.getoption("--state", default=None)
    breakpoint()
    return profile_data_generation(output_dir)(generate_current_population_survey)(year=year, state=state)


@pytest.fixture(scope="session")
def census_dataset(output_dir, request):
    year = int(request.config.getoption("--year", default=2020))
    state = request.config.getoption("--state", default=None)
    return profile_data_generation(output_dir)(generate_decennial_census)(year=year, state=state)


@pytest.fixture(scope="session")
def ssa_dataset(output_dir, request):
    year = int(request.config.getoption("--year", default=2020))
    return profile_data_generation(output_dir)(generate_social_security)(year=year)


@pytest.fixture(scope="session")
def taxes_1040_dataset(output_dir, request):
    year = int(request.config.getoption("--year", default=2020))
    state = request.config.getoption("--state", default=None)
    return profile_data_generation(output_dir)(generate_taxes_1040)(year=year, state=state)


@pytest.fixture(scope="session")
def taxes_w2_and_1099_dataset(output_dir, request):
    year = int(request.config.getoption("--year", default=2020))
    state = request.config.getoption("--state", default=None)
    return profile_data_generation(output_dir)(generate_taxes_w2_and_1099)(year=year, state=state)


@pytest.fixture(scope="session")
def wic_dataset(output_dir, request):
    year = int(request.config.getoption("--year", default=2020))
    state = request.config.getoption("--state", default=None)
    return profile_data_generation(output_dir)(generate_women_infants_and_children)(year=year, state=state)


####################
# Helper Functions #
####################
def profile_data_generation(output_dir: Path) -> Callable[..., pd.DataFrame]:
    """Decorator to profile a function's time and memory usage."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            start_time = time.time()
            mem_before = memory_usage(interval=1, timeout=1)
            df = func(*args, **kwargs)
            mem_after = memory_usage(interval=1, timeout=1)
            end_time = time.time()
            resources = pd.DataFrame(
                {
                    "time_s": end_time - start_time,
                    "memory_gb": (mem_after[0] - mem_before[0]) / 1024,
                },
                index=[0],
            )
            filename = f"{func.__name__}_resources.csv"
            output_path = os.path.join(output_dir, filename)
            resources.to_csv(output_path, index=False)
            return df

        return wrapper

    return decorator