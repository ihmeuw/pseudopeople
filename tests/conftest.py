from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Generator
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from _pytest.config import Config, argparsing
from _pytest.logging import LogCaptureFixture
from _pytest.python import Function
from loguru import logger
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from tests.integration.conftest import CELL_PROBABILITY


def pytest_addoption(parser: argparsing.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--release", action="store_true", default=False, help="run release tests")
    parser.addoption(
        "--limit",
        action="store",
        default=-1,
        type=int,
        help="Maximum number of parameterized tests to run",
    )


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: Config, items: list[Function]) -> None:
    skip_release = pytest.mark.skip(reason="need --release to run")
    if not config.getoption("--release"):
        for item in items:
           if 'test_release.py' in item.keywords or 'test_runner.py' in item.keywords:
               item.add_marker(skip_release)

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        # Automatically tag all tests in the tests/integration dir as slow
        if item.parent and Path(item.parent.path).parent.stem == "integration":
            item.add_marker(pytest.mark.slow)
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

    # Limit the number of permutations of parametrised tests to run.
    limit = config.getoption("--limit")
    if limit > 0:
        tests_by_name = {item.name: item for item in items}
        # Add the name of parametrized base tests to this list.
        tests_to_skip_parametrize = ["test_noise_order"]

        for base_name in tests_to_skip_parametrize:
            to_skip = [t for n, t in tests_by_name.items() if base_name in n][limit:]
            for t in to_skip:
                t.add_marker("skip")


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator[LogCaptureFixture, None, None]:
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="session")
def output_directory() -> Path:
    v_v_path = Path(os.path.dirname(__file__)) / "v_and_v_output"
    return v_v_path


@pytest.fixture(scope="session")
def fuzzy_checker(output_directory: Path) -> Generator[FuzzyChecker, None, None]:
    checker = FuzzyChecker()

    yield checker

    checker.save_diagnostic_output(output_directory)


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Returns a custom configuration dict to be used in noising"""
    ROW_PROBABILITY = 0.05
    config = get_configuration().to_dict()  # default config

    # Increase row noise probabilities to 5% and column cell_probabilities to 25%
    for dataset_name in config:
        dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
        config[dataset_schema.name][Keys.ROW_NOISE] = {
            noise_type.name: {
                Keys.ROW_PROBABILITY: ROW_PROBABILITY,
            }
            for noise_type in dataset_schema.row_noise_types
            if noise_type != NOISE_TYPES.duplicate_with_guardian
        }
        for col in [c for c in dataset_schema.columns if c.noise_types]:
            config[dataset_name][Keys.COLUMN_NOISE][col.name] = {
                noise_type.name: {
                    Keys.CELL_PROBABILITY: CELL_PROBABILITY,
                }
                for noise_type in col.noise_types
            }

    # FIXME: Remove when record_id is added as the truth deck for datasets.
    # For integration tests, we will NOT duplicate rows with guardian duplication.
    # This is because we want to be able to compare the noised and unnoised data
    # and a big assumption we make is that simulant_id and household_id are the
    # truth decks in our datasets.
    config[DATASET_SCHEMAS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.duplicate_with_guardian.name
    ] = {
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0.0,
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0.0,
    }
    # Update SSA dataset to noise 'ssn' but NOT noise 'ssa_event_type' since that
    # will be used as an identifier along with simulant_id
    # TODO: Noise ssa_event_type when record IDs are implemented (MIC-4039)
    config[DATASET_SCHEMAS.ssa.name][Keys.COLUMN_NOISE][COLUMNS.ssa_event_type.name] = {
        noise_type.name: {
            Keys.CELL_PROBABILITY: 0,
        }
        for noise_type in COLUMNS.ssa_event_type.noise_types
    }
    return config
