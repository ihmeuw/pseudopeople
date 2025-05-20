from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from _pytest.fixtures import FixtureRequest
from layered_config_tree import LayeredConfigTree
from pytest_check import check
from pytest_mock import MockerFixture
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.entities import NO_NOISE
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.dataset import Dataset
from pseudopeople.dtypes import DtypeNames
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.filter import get_data_filters
from pseudopeople.interface import (
    generate_social_security,
    get_dataset_filepaths,
    validate_source_compatibility,
)
from pseudopeople.loader import load_standard_dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_functions import merge_dependents_and_guardians
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from pseudopeople.utilities import (
    DASK_ENGINE,
    get_engine_from_string,
    to_string,
    update_seed,
)
from tests.constants import TOKENS_PER_STRING_MAPPER
from tests.integration.conftest import SEED, get_common_datasets
from tests.integration.release.conftest import (
    DATASET_ARG_TO_FULL_NAME_MAPPER,
    RI_FILEPATH,
)
from tests.integration.release.utilities import (
    run_do_not_respond_tests,
    run_guardian_duplication_tests,
    run_omit_row_tests,
)
from tests.utilities import (
    get_single_noise_type_config,
    initialize_dataset_with_sample,
    run_column_noising_tests,
)

if TYPE_CHECKING:
    import dask.dataframe as dd


ROW_TEST_FUNCTIONS = {
    "omit_row": run_omit_row_tests,
    "do_not_respond": run_do_not_respond_tests,
    "duplicate_with_guardian": run_guardian_duplication_tests,
}


# taken from Dataset._clean_input_data
def clean_input_data(dataset: Dataset) -> None:
    for col in dataset.dataset_schema.columns:
        # Coerce empty strings to nans
        dataset.data[col.name] = dataset.data[col.name].replace("", np.nan)

        if (
            dataset.data[col.name].dtype.name == "category"
            and col.dtype_name == DtypeNames.OBJECT
        ):
            # We made some columns in the pseudopeople input categorical
            # purely as a kind of DIY compression.
            # TODO: Determine whether this is benefitting us after
            # the switch to Parquet.
            dataset.data[col.name] = to_string(dataset.data[col.name])


def test_full_release_noising(
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

    for dataset in datasets:
        clean_input_data(dataset)

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

        if isinstance(noise_type, ColumnNoiseType):
            for column in prenoised_dataframes[0].columns:
                if config.has_noise_type(dataset_schema.name, noise_type.name, column):
                    # don't noise ssa_event_type because it's used as an identifier column
                    # along with simulant id
                    # TODO: Noise ssa_event_type when record IDs are implemented (MIC-4039)
                    if column == COLUMNS.ssa_event_type.name:
                        pass
                    else:
                        for dataset in datasets:
                            # noise datasets in place
                            noise_type(dataset, config, column)
                            with check:
                                assert dataset.missingness.equals(
                                    dataset.is_missing(dataset.data)
                                )
                        run_column_noising_test(
                            prenoised_dataframes,
                            datasets,
                            config,
                            full_dataset_name,
                            noise_type.name,
                            column,
                            fuzzy_checker,
                        )


def run_column_noising_test(
    prenoised_dataframes: list[pd.DataFrame],
    noised_datasets: list[Dataset],
    config: NoiseConfiguration,
    dataset_name: str,
    noise_type: str,
    column: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)

    numerator = 0
    denominator = 0
    expected_noise_level = 0

    # Validate column noise level
    expected_config_noise = config.get_cell_probability(dataset_name, noise_type, column)
    includes_token_noising = config.has_parameter(
        dataset_name, noise_type, Keys.TOKEN_PROBABILITY, column
    ) or config.has_parameter(
        dataset_name, noise_type, Keys.ZIPCODE_DIGIT_PROBABILITIES, column
    )

    for prenoised_dataframe, noised_dataset in zip(prenoised_dataframes, noised_datasets):
        shared_noised, shared_prenoised, shared_idx = get_common_datasets(
            dataset_schema, prenoised_dataframe, noised_dataset.data
        )

        # Check that originally missing data remained missing
        originally_missing_idx = shared_prenoised.index[
            shared_prenoised.reset_index()[column].isna()
        ]
        with check:
            assert shared_noised.loc[originally_missing_idx, column].isna().all()

        # Check for noising where applicable
        to_compare_idx = shared_idx.difference(originally_missing_idx)
        if len(to_compare_idx) == 0:
            continue

        # make sure dtypes match when comparing prenoised and noised values
        # adapted from utilities.coerce_dtypes
        if shared_prenoised[column].dtype.name != shared_noised[column].dtype.name:
            if shared_noised[column].dtype.name == DtypeNames.OBJECT:
                shared_prenoised[column] = to_string(shared_prenoised[column])
            else:
                # mypy doesn't like using a variable as an argument to astype
                shared_prenoised[column] = shared_prenoised[column].astype(shared_noised[column].dtype.name)  # type: ignore [call-overload]

        prenoised_values = shared_prenoised.loc[to_compare_idx, column].values
        noised_values = shared_noised.loc[to_compare_idx, column].values

        different_check: npt.NDArray[np.bool_] = np.array(prenoised_values != noised_values)
        noise_level = different_check.sum()

        if includes_token_noising:
            if noise_type == NOISE_TYPES.write_wrong_zipcode_digits.name:
                token_probability: list[
                    float
                ] | int | float = config.get_zipcode_digit_probabilities(dataset_name, column)
            else:
                token_probability = config.get_token_probability(
                    dataset_name, noise_type, column
                )

            # Get number of tokens per string to calculate expected proportion
            tokens_per_string_getter: Callable[
                ..., pd.Series[int] | int
            ] = TOKENS_PER_STRING_MAPPER.get(noise_type, lambda x: x.astype(str).str.len())
            tokens_per_string: pd.Series[int] | int = tokens_per_string_getter(
                shared_prenoised.loc[to_compare_idx, column]
            )
            # Calculate probability no token is noised
            if isinstance(token_probability, list):
                # Calculate write wrong zipcode average digits probability any token is noise
                avg_probability_any_token_noised = 1 - math.prod(
                    [1 - p for p in token_probability]
                )
            else:
                with check:
                    assert isinstance(tokens_per_string, pd.Series)
                avg_probability_any_token_noised = (
                    1 - (1 - token_probability) ** tokens_per_string
                ).mean()

            # This is accumulating not_noised over all noise types
            expected_noise = avg_probability_any_token_noised * expected_config_noise
        else:
            expected_noise = expected_config_noise

        num_eligible = len(to_compare_idx)
        # we sometimes copy the same column value from a household member so we only want
        # to consider individuals who ended up with a different value as a result of this noising
        can_silently_noise = noise_type == "copy_from_household_member" and (
            column == "age" or column == "date_of_birth"
        )
        if can_silently_noise:
            try:
                num_sims_with_silent_noising = sum(
                    shared_prenoised.loc[to_compare_idx, column].astype(float)
                    == shared_prenoised.loc[to_compare_idx, f"copy_{column}"].astype(float)
                )
            except:
                num_sims_with_silent_noising = sum(
                    shared_prenoised.loc[to_compare_idx, column].astype(str)
                    == shared_prenoised.loc[to_compare_idx, f"copy_{column}"].astype(str)
                )
            num_eligible -= num_sims_with_silent_noising

        numerator += noise_level
        denominator += num_eligible
        expected_noise_level += expected_noise * num_eligible

    fuzzy_checker.fuzzy_assert_proportion(
        name=noise_type,
        observed_numerator=numerator,
        observed_denominator=denominator,
        target_proportion=expected_noise_level / denominator,
        name_additional=f"{dataset_name}_{column}_{noise_type}",
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
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    check_noised, check_original, _ = get_common_datasets(
        dataset_schema, original.data, noised_data
    )
    assert (
        (
            check_original.reset_index()[unnoised_id_cols]
            == check_noised.reset_index()[unnoised_id_cols]
        )
        .all()
        .all()
    )
