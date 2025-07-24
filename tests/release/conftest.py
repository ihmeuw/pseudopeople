import functools
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from memory_profiler import memory_usage  # type: ignore

from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)

DATASET_GENERATION_FUNCS: dict[str, Callable[..., Any]] = {
    "census": generate_decennial_census,
    "acs": generate_american_community_survey,
    "cps": generate_current_population_survey,
    "ssa": generate_social_security,
    "tax_w2_1099": generate_taxes_w2_and_1099,
    "wic": generate_women_infants_and_children,
    "tax_1040": generate_taxes_1040,
}

CLI_DEFAULT_DATASET = "acs"
CLI_DEFAULT_POP = "sample"
CLI_DEFAULT_YEAR = 2020
CLI_DEFAULT_STATE = None
CLI_DEFAULT_ENGINE = "pandas"
FULL_USA_FILEPATH = "/mnt/team/simulation_science/pub/models/vivarium_census_prl_synth_pop/results/release_02_yellow/full_data/united_states_of_america/2023_08_21_16_35_27/final_results/2023_08_31_15_58_01/pseudopeople_simulated_population_usa_2_0_0"
RI_FILEPATH = "/mnt/team/simulation_science/pub/models/vivarium_census_prl_synth_pop/results/release_02_yellow/full_data/united_states_of_america/2023_08_21_16_35_27/final_results/2023_08_31_15_58_01/states/pseudopeople_simulated_population_rhode_island_2_0_0"
SOURCE_MAPPER = {"usa": FULL_USA_FILEPATH, "ri": RI_FILEPATH, "sample": None}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dataset",
        action="store",
        default=CLI_DEFAULT_DATASET,
        help="The dataset to generate. Options are 'census', 'acs', 'cps', 'ssa', 'tax_w2_1099', 'wic', and 'tax_1040'. No argument will default to acs.",
    )
    parser.addoption(
        "--population",
        action="store",
        default=CLI_DEFAULT_POP,
        help="The simulated population to generate. Options are 'USA', 'RI', and 'sample'. sample will generate very small sample data.",
    )
    parser.addoption(
        "--year",
        action="store",
        default=CLI_DEFAULT_YEAR,
        help="The year to subset our data to.",
    )
    parser.addoption(
        "--state",
        action="store",
        default=CLI_DEFAULT_STATE,
        help="The state to subset our data to (if using full USA population) using 2-letter abbreviations. No argument means no subsetting will be done.",
    )
    parser.addoption(
        "--engine",
        action="store",
        default=CLI_DEFAULT_ENGINE,
        help="The engine used to generate data. Options are 'pandas' and 'dask'.",
    )


############
# Fixtures #
############
@pytest.fixture(scope="session")
def release_output_dir() -> Path:
    # TODO: [MIC-5522] define correct output dir
    # output_dir = os.environ.get("PSP_TEST_OUTPUT_DIR")
    output_dir_name = (
        "/mnt/team/simulation_science/priv/engineering/pseudopeople_release_testing"
    )
    # if not output_dir_name:
    #     raise ValueError("PSP_TEST_OUTPUT_DIR environment variable not set")
    output_dir = Path(output_dir_name) / f"{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


@pytest.fixture(scope="session")
def dataset(release_output_dir: Path, request: pytest.FixtureRequest) -> pd.DataFrame:
    _, dataset_func, source, year, state, engine = parse_dataset_params(request)

    kwargs = {
        "source": source,
        "year": year,
        "engine": engine,
    }
    if dataset_func != generate_social_security:
        kwargs["state"] = state
    return profile_data_generation(release_output_dir)(dataset_func)(**kwargs)


####################
# Helper Functions #
####################
def profile_data_generation(output_dir: Path) -> Callable[..., Callable[..., pd.DataFrame]]:
    """Decorator to profile a function's time and memory usage."""
    # TODO: [MIC-5522] properly setup profiling
    def decorator(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
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


def parse_dataset_params(
    request: pytest.FixtureRequest,
) -> tuple[str | int | Callable[..., pd.DataFrame] | None, ...]:
    dataset_name = request.config.getoption("--dataset", default=CLI_DEFAULT_DATASET)
    try:
        dataset_func = DATASET_GENERATION_FUNCS[dataset_name]
    except KeyError:
        raise ValueError(
            f"{dataset_name} is not a valid dataset. Possible datasets are {','.join(DATASET_GENERATION_FUNCS.keys())}"
        )

    population = request.config.getoption("--population", default=CLI_DEFAULT_POP)
    try:
        source = SOURCE_MAPPER[population.lower()]
    except KeyError:
        raise ValueError(
            f"population must be one of 'USA', 'RI', or 'sample'. You passed in '{population}'."
        )

    year = int(request.config.getoption("--year", default=CLI_DEFAULT_YEAR))
    state = request.config.getoption("--state", default=CLI_DEFAULT_STATE)
    engine = request.config.getoption("--engine", default=CLI_DEFAULT_ENGINE)

    return dataset_name, dataset_func, source, year, state, engine
