from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from _pytest.fixtures import FixtureRequest
from pytest_mock import MockerFixture
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import get_configuration
from pseudopeople.configuration.entities import NO_NOISE
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.dataset import Dataset
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.filter import get_data_filters
from pseudopeople.interface import (
    generate_social_security,
    get_dataset_filepaths,
    validate_source_compatibility,
)
from pseudopeople.loader import load_standard_dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from pseudopeople.utilities import DASK_ENGINE, get_engine_from_string, update_seed
from tests.integration.conftest import SEED, _get_common_datasets
from tests.integration.release.conftest import DATASET_ARG_TO_FULL_NAME_MAPPER
from tests.integration.release.utilities import (
    run_do_not_respond_tests,
    run_guardian_duplication_tests,
    run_omit_row_tests,
)
from tests.utilities import initialize_dataset_with_sample, run_column_noising_tests

if TYPE_CHECKING:
    import dask.dataframe as dd


ROW_TEST_FUNCTIONS = {
    "omit_row": run_omit_row_tests,
    "do_not_respond": run_do_not_respond_tests,
    "duplicate_with_guardian": run_guardian_duplication_tests,
}


def test_release_row_noising(
    dataset_params: tuple[
        str,
        Callable[..., pd.DataFrame],
        Path | str | None,
        int | None,
        str | None,
        Literal["pandas", "dask"],
    ],
    fuzzy_checker: FuzzyChecker,
) -> None:
    dataset_name, _, source, year, state, engine_name = dataset_params
    full_dataset_name = DATASET_ARG_TO_FULL_NAME_MAPPER[dataset_name]
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(full_dataset_name)
    config = get_configuration()

    # update parameters
    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    elif isinstance(source, str) or isinstance(source, Path):
        source = Path(source)
        validate_source_compatibility(source, dataset_schema)

    engine = get_engine_from_string(engine_name)

    data_file_paths = get_dataset_filepaths(Path(source), dataset_schema.name)
    filters = get_data_filters(dataset_schema, year, state)
    unnoised_data: list[pd.DataFrame | dd.DataFrame] = [
        load_standard_dataset(path, filters, engine) for path in data_file_paths
    ]

    if engine == DASK_ENGINE:
        # TODO: [MIC-5960] move this compute to later in the code
        dataset_data: list[pd.DataFrame] = [data.compute() for data in unnoised_data if len(data) != 0]  # type: ignore [operator]
    else:
        dataset_data = [data for data in unnoised_data if len(data) != 0]  # type: ignore [misc]

    seed = update_seed(SEED, year)
    datasets: list[Dataset] = [
        Dataset(dataset_schema, data, f"{seed}_{i}") for i, data in enumerate(dataset_data)
    ]

    for noise_type in NOISE_TYPES:
        prenoised_dataframes: list[pd.DataFrame] = [
            dataset.data.copy() for dataset in datasets
        ]
        if isinstance(noise_type, RowNoiseType):
            if config.has_noise_type(dataset_schema.name, noise_type.name):
                for dataset in datasets:
                    # noise datasets in place
                    noise_type(dataset, config)
                test_function = ROW_TEST_FUNCTIONS[noise_type.name]
                test_function(
                    prenoised_dataframes, datasets, config, full_dataset_name, fuzzy_checker
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
