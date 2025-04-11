from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.entities import NO_NOISE
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.dataset import Dataset
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.interface import generate_decennial_census, generate_social_security
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_functions import merge_dependents_and_guardians
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from tests.integration.conftest import SEED, _get_common_datasets
from tests.integration.release.conftest import DATASET_ARG_TO_FULL_NAME_MAPPER
from tests.integration.release.utilities import (
    run_do_not_respond_tests,
    run_guardian_duplication_tests,
    run_omit_row_tests,
)
from tests.utilities import initialize_dataset_with_sample, run_column_noising_tests

ROW_TEST_FUNCTIONS = {
    "omit_row": run_omit_row_tests,
    "do_not_respond": run_do_not_respond_tests,
    "duplicate_with_guardian": run_guardian_duplication_tests,
}


def test_release_runs(
    dataset_params: tuple[
        str,
        Callable[..., pd.DataFrame],
        str | None,
        int | None,
        str | None,
        Literal["pandas", "dask"],
    ],
    fuzzy_checker: FuzzyChecker,
    mocker: MockerFixture,
) -> None:
    # keep all columns when generating unnoised data because some of them are used in testing
    mocker.patch(
        "pseudopeople.dataset.Dataset.drop_non_schema_columns", side_effect=lambda df, _: df
    )

    # create unnoised dataset
    dataset_name, dataset_func, source, year, state, engine = dataset_params
    unnoised_data_kwargs = {
        "source": source,
        "config": NO_NOISE,
        "year": year,
        "engine": engine,
    }
    if dataset_func != generate_social_security:
        unnoised_data_kwargs["state"] = state
    unnoised_data = dataset_func(**unnoised_data_kwargs)

    # In our standard noising process, i.e. when noising a shard of data, we
    # 1) clean and reformat the data, 2) noise the data, and 3) do some post-processing.
    # We're replicating steps 1 and 2 in this test and skipping 3.
    full_dataset_name = DATASET_ARG_TO_FULL_NAME_MAPPER[dataset_name]
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(full_dataset_name)
    dataset = Dataset(dataset_schema, unnoised_data, SEED)
    # don't unnecessarily keep in memory
    del unnoised_data
    dataset._clean_input_data()
    # convert datetime columns to datetime types for _reformat_dates_for_noising
    # because the post-processing that occured in generating the unnoised data
    # in step 3 mentioned above converts these columns to object dtypes
    for col in [COLUMNS.dob.name, COLUMNS.ssa_event_date.name]:
        if col in dataset.data:
            dataset.data[col] = pd.to_datetime(
                dataset.data[col], format=dataset_schema.date_format
            )
            copy_col = "copy_" + col
            if copy_col in dataset.data:
                dataset.data[copy_col] = pd.to_datetime(
                    dataset.data[copy_col], format=dataset_schema.date_format
                )
    dataset._reformat_dates_for_noising()

    config = get_configuration()

    for noise_type in NOISE_TYPES:
        original_data = dataset.data.copy()
        if isinstance(noise_type, RowNoiseType):
            if config.has_noise_type(dataset.dataset_schema.name, noise_type.name):
                noise_type(dataset, config)
                test_function = ROW_TEST_FUNCTIONS[noise_type.name]
                test_function(
                    original_data, dataset.data, config, full_dataset_name, fuzzy_checker
                )


def test_column_noising(
    unnoised_dataset: Dataset,
    noised_data: pd.DataFrame,
    config: dict[str, Any],
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that columns are noised as expected"""
    check_noised, check_original, shared_idx = _get_common_datasets(
        unnoised_dataset, noised_data
    )

    run_column_noising_tests(
        dataset_name, config, fuzzy_checker, check_noised, check_original, shared_idx
    )


def test_column_dtypes(
    unnoised_dataset: Dataset,
    noised_data: pd.DataFrame,
    dataset_name: str,
    config: dict[str, Any],
) -> None:
    """Tests that column dtypes are as expected"""
    for col_name in noised_data.columns:
        col = COLUMNS.get_column(col_name)
        expected_dtype = col.dtype_name
        if expected_dtype == np.dtype(object):
            # str dtype is 'object'
            # Check that they are actually strings and not some other
            # type of object.
            # mypy wants typed type_function to pass into apply but doesn't
            # accept type as an output
            type_function: Callable[..., Any] = lambda x: type(x)
            actual_types = noised_data[col.name].dropna().apply(type_function)
            assert (actual_types == str).all(), actual_types.unique()
        assert noised_data[col.name].dtype == expected_dtype


def test_unnoised_id_cols(dataset_name: str, request: FixtureRequest) -> None:
    """Tests that all datasets retain unnoised simulant_id and household_id
    (except for SSA which does not include household_id)
    """
    unnoised_id_cols = [COLUMNS.simulant_id.name]
    if dataset_name != DATASET_SCHEMAS.ssa.name:
        unnoised_id_cols.append(COLUMNS.household_id.name)
    original = initialize_dataset_with_sample(dataset_name)
    noised_data = request.getfixturevalue("noised_data")
    check_noised, check_original, _ = _get_common_datasets(original, noised_data)
    assert (
        (
            check_original.reset_index()[unnoised_id_cols]
            == check_noised.reset_index()[unnoised_id_cols]
        )
        .all()
        .all()
    )


def test_dataset_missingness(
    dataset_params: tuple[
        str,
        Callable[..., pd.DataFrame],
        str | None,
        int | None,
        str | None,
        Literal["pandas", "dask"],
    ],
    dataset_name: str,
    mocker: MockerFixture,
) -> None:
    """Tests that missingness is accurate with dataset.data."""
    mocker.patch(
        "pseudopeople.dataset.Dataset.drop_non_schema_columns", side_effect=lambda df, _: df
    )

    # create unnoised dataset
    _, dataset_func, source, year, state, engine = dataset_params
    kwargs = {
        "source": source,
        "config": NO_NOISE,
        "year": year,
        "engine": engine,
    }
    if dataset_func != generate_social_security:
        kwargs["state"] = state
    unnoised_data = dataset_func(**kwargs)

    # In our standard noising process, i.e. when noising a shard of data, we
    # 1) clean and reformat the data, 2) noise the data, and 3) do some post-processing.
    # We're replicating steps 1 and 2 in this test and skipping 3.
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    dataset = Dataset(dataset_schema, unnoised_data, SEED)
    dataset._clean_input_data()
    # convert datetime columns to datetime types for _reformat_dates_for_noising
    # because the post-processing that occured in generating the unnoised data
    # in step 3 mentioned above converts these columns to object dtypes
    for col in [COLUMNS.dob.name, COLUMNS.ssa_event_date.name]:
        if col in dataset.data:
            dataset.data[col] = pd.to_datetime(dataset.data[col])
            dataset.data["copy_" + col] = pd.to_datetime(dataset.data["copy_" + col])
    dataset._reformat_dates_for_noising()
    config = get_configuration()

    # NOTE: This is recreating Dataset._noise_dataset but adding assertions for missingness
    for noise_type in NOISE_TYPES:
        if isinstance(noise_type, RowNoiseType):
            if config.has_noise_type(dataset.dataset_schema.name, noise_type.name):
                noise_type(dataset, config)
                # Check missingness is synced with data
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))
        if isinstance(noise_type, ColumnNoiseType):
            for column in dataset.data.columns:
                if config.has_noise_type(
                    dataset.dataset_schema.name, noise_type.name, column
                ):
                    noise_type(dataset, config, column)
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))
