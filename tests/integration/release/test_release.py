from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask.dataframe as dd
import numpy as np
import numpy.typing as npt
import pandas as pd
from _pytest.fixtures import FixtureRequest
from dask.distributed import LocalCluster
from pytest_check import check
from pytest_mock import MockerFixture
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.entities import NO_NOISE
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.dataset import Dataset, clean_input_data, reformat_dates_for_noising
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
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS, Column
from pseudopeople.utilities import (
    DASK_ENGINE,
    coerce_dtypes,
    get_engine_from_string,
    parse_dates,
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
    fuzzy_check_omit_row_counts,
    get_high_noise_config,
    get_missingness_data,
    get_noised_columns,
    get_noised_data,
    get_omit_row_counts,
    get_passing_row_counts,
    get_prenoised_columns,
    get_prenoised_data,
    run_do_not_respond_tests,
    run_guardian_duplication_tests,
    run_omit_row_tests,
)
from tests.utilities import initialize_dataset_with_sample

ROW_TEST_FUNCTIONS = {
    "omit_row": run_omit_row_tests,
    "do_not_respond": run_do_not_respond_tests,
    "duplicate_with_guardian": run_guardian_duplication_tests,
}

ROW_COUNT_FUNCTIONS = {
    "omit_row": get_omit_row_counts,
    # TODO: define all ROW_COUNT_FUNCTIONS functions
    "do_not_respond": get_passing_row_counts,
    "duplicate_with_guardian": get_passing_row_counts,
}

ROW_FUZZY_CHECK_FUNCTIONS = {
    "omit_row": fuzzy_check_omit_row_counts,
    # TODO: define all ROW_FUZZY_CHECK functions
    "do_not_respond": fuzzy_check_omit_row_counts,
    "duplicate_with_guardian": fuzzy_check_omit_row_counts,
}


def test_full_release_noising(
    dataset_params: tuple[
        str,
        Callable[..., pd.DataFrame],
        Path | str | None,
        int | None,
        str | None,
        Literal["pandas", "dask"],
        str,
    ],
    fuzzy_checker: FuzzyChecker,
) -> None:
    dataset_name, _, source, year, state, engine_name, noise_level = dataset_params
    full_dataset_name = DATASET_ARG_TO_FULL_NAME_MAPPER[dataset_name]
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(full_dataset_name)
    if noise_level == "default":
        config = get_configuration()
    elif noise_level == "high":
        config = get_high_noise_config(full_dataset_name)
    else:
        raise ValueError(
            f"noise level must be 'default' or 'high', but {noise_level} was passed instead."
        )

    # update parameters
    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    elif isinstance(source, str) or isinstance(source, Path):
        source = Path(source)
        validate_source_compatibility(source, dataset_schema)

    source = Path("/ihme/homes/hjafari/ppl_parquet_data/tax_1040/")

    engine = get_engine_from_string(engine_name)

    if engine == DASK_ENGINE:
        cluster = LocalCluster(  # type: ignore[no-untyped-call]
            n_workers=int(os.environ["SLURM_CPUS_ON_NODE"]),
            threads_per_worker=1,
            memory_limit=(
                # Per worker!
                int(os.environ["SLURM_MEM_PER_NODE"])
                / int(os.environ["SLURM_CPUS_ON_NODE"])
            )
            * 1_000
            * 1_000,  # Dask uses bytes, Slurm reports in megabytes.
        )
        client = cluster.get_client()  # type: ignore[no-untyped-call]

        data_directory_path = source / dataset_schema.name
        filters = get_data_filters(dataset_schema, year, state)
        data = load_standard_dataset(data_directory_path, filters, engine, is_file=False)
        assert isinstance(data, dd.DataFrame)

        seed = update_seed(SEED, year)

        def make_wide_dataframe(data_: pd.DataFrame) -> pd.DataFrame:
            clean_input_data(dataset_schema, data_)
            reformat_dates_for_noising(dataset_schema, data_)
            missingness = Dataset.is_missing(data_)
            missingness.columns = [f"{col}_missingness" for col in missingness.columns]
            prenoised = data_.copy()
            prenoised.columns = [f"{col}_prenoised" for col in prenoised.columns]
            data_ = pd.concat([data_, missingness, prenoised], axis=1)
            return data_

        def wide_to_dataset(
            wide_data: pd.DataFrame, partition_info: dict[str, Any]
        ) -> Dataset:
            # noised data becomes prenoised data of the next noise type
            data_to_noise = get_noised_data(wide_data)
            dataset = Dataset(
                dataset_schema,
                data_to_noise,
                f"{seed}_{partition_info['number'] if partition_info is not None else 1}",
            )
            return dataset

        def dataset_to_wide(dataset: Dataset, old_wide_data: pd.DataFrame) -> pd.DataFrame:
            noised = dataset.data
            missingness = Dataset.is_missing(noised)
            missingness.columns = [f"{col}_missingness" for col in missingness.columns]
            prenoised = get_noised_data(old_wide_data)
            prenoised.columns = [f"{col}_prenoised" for col in prenoised.columns]
            data = pd.concat([noised, missingness, prenoised], axis=1)
            return data

        def apply_row_noise_type(
            data_: pd.DataFrame,
            partition_info: dict[str, Any],
            noise_type: RowNoiseType,
            config: NoiseConfiguration,
        ) -> pd.DataFrame:
            dataset = wide_to_dataset(data_, partition_info)
            noise_type(dataset, config)
            data_ = dataset_to_wide(dataset, data_)
            return data_

        def apply_column_noise_type(
            data_: pd.DataFrame,
            column: Column,
            partition_info: dict[str, Any],
            noise_type: ColumnNoiseType,
            config: NoiseConfiguration,
        ) -> pd.DataFrame:
            dataset = wide_to_dataset(data_, partition_info)
            noise_type(dataset, config, column.name)
            data_ = dataset_to_wide(dataset, data_)
            return data_

        def unnoise_data(data_: pd.DataFrame) -> pd.DataFrame:
            data_[get_noised_columns(data_)] = data_[get_prenoised_columns(data_)]
            return data_

        def drop_extra_data(data_: pd.DataFrame) -> pd.DataFrame:
            """Removed prenoised and missingness columns from data."""
            return data_[get_noised_columns(data_)]

        def aggregate_dtype_info(results: pd.DataFrame) -> pd.DataFrame:
            """Aggregates results across partitions to get one row per column."""
            return results.groupby("column").agg(
                {
                    "is_expected_dtype": "all",  # Logical AND across partitions
                    "object_column_types": lambda x: ",".join(
                        set(filter(None, x))
                    ),  # Combine non-None values
                }
            )

        # create wide data metadata
        wide_data_columns = (
            [col for col in data.columns]
            + [f"{col}_missingness" for col in data.columns]
            + [f"{col}_prenoised" for col in data.columns]
        )
        original_dtypes = dict(data.dtypes)
        missingness_dtypes = {
            f"{col.name}_missingness": bool for col in dataset_schema.columns
        }
        prenoised_dtypes = {f"{col}_prenoised": dtype for col, dtype in data.dtypes.items()}
        all_dtypes = {**original_dtypes, **missingness_dtypes, **prenoised_dtypes}
        wide_data_meta = pd.DataFrame(columns=wide_data_columns).astype(all_dtypes)

        data = data.map_partitions(make_wide_dataframe, meta=wide_data_meta)  # type: ignore[no-untyped-call]
        noised_metadata = list(zip(data.columns, data.dtypes))
        count_metadata = pd.DataFrame(
            columns=["numerator", "columns_are_different", "dtypes_are_different"], dtype=int
        )

        for noise_type in NOISE_TYPES:
            if isinstance(noise_type, RowNoiseType):
                if config.has_noise_type(dataset_schema.name, noise_type.name):
                    data = data.persist()
                    num_prenoised_rows = len(data)

                    data = data.map_partitions(
                        apply_row_noise_type,
                        noise_type=noise_type,
                        config=config,
                        meta=wide_data_meta,
                    ).persist()

                    # get counts
                    # TODO: define all ROW_COUNT_FUNCTIONS functions
                    count_function = ROW_COUNT_FUNCTIONS[noise_type.name]
                    counts = data.map_partitions(
                        count_function,
                        meta=count_metadata,
                    )
                    total_counts = counts.sum().compute()

                    numerator = total_counts["numerator"]
                    # use counts to make checks
                    # TODO: define all ROW_FUZZY_CHECK_FUNCTIONS functions
                    with check:
                        assert total_counts["columns_are_different"] == 0
                    with check:
                        assert total_counts["dtypes_are_different"] == 0
                    fuzzy_check_function = ROW_FUZZY_CHECK_FUNCTIONS[noise_type.name]
                    fuzzy_check_function(
                        numerator,
                        num_prenoised_rows,
                        config,
                        full_dataset_name,
                        fuzzy_checker,
                    )

            elif isinstance(noise_type, ColumnNoiseType):
                for column in dataset_schema.columns:
                    if config.has_noise_type(
                        dataset_schema.name, noise_type.name, column.name
                    ):
                        # don't noise ssa_event_type because it's used as an identifier column
                        # along with simulant id
                        # TODO: Noise ssa_event_type when record IDs are implemented (MIC-4039)
                        if column == COLUMNS.ssa_event_type:
                            continue
                        data = data.map_partitions(
                            apply_column_noise_type,
                            column=column,
                            noise_type=noise_type,
                            config=config,
                            meta=noised_metadata,
                        ).persist()

                        missingness_correct = data.map_partitions(
                            lambda data_: pd.Series(
                                (
                                    get_missingness_data(data_).values
                                    == Dataset.is_missing(get_noised_data(data_)).values
                                ).all()
                            )
                        )
                        with check:
                            assert missingness_correct.all().compute()

                        counts = data.map_partitions(
                            lambda data_: get_column_noising_counts(
                                data_,
                                config,
                                dataset_schema.name,
                                noise_type.name,
                                column.name,
                            ),
                            meta=pd.DataFrame(
                                columns=[
                                    "numerator",
                                    "denominator",
                                    "expected_numerator",
                                    "missing_data_not_missing",
                                ],
                                dtype=int,
                            ),
                        )
                        total_counts = counts.sum().compute()
                        with check:
                            assert total_counts["missing_data_not_missing"] == 0
                        numerator = total_counts["numerator"]
                        denominator = total_counts["denominator"]
                        # we can get no eligible rows for unit number in acs data because it
                        # it's a sparse column and acs is small
                        if column == COLUMNS.unit_number and denominator == 0:
                            continue
                        expected_numerator = total_counts["expected_numerator"]
                        with check:
                            fuzzy_checker.fuzzy_assert_proportion(
                                name=noise_type.name,
                                observed_numerator=numerator,
                                observed_denominator=denominator,
                                target_proportion=expected_numerator / denominator,
                                name_additional=f"{dataset_name}_{column}_{noise_type}",
                            )

            if noise_type == NOISE_TYPES.duplicate_with_guardian:
                # noising after duplicate_with_guardian should be done on prenoised data
                # since it duplicates simulant ID which must be unique to be used as an identifier
                # TODO: Noise duplicate_with_guardian normally when record IDs
                # are implemented (MIC-4039)
                data = data.map_partitions(unnoise_data)

        # remove prenoised and missingness columns from data
        data = data.map_partitions(drop_extra_data)
        # post-processing tests on final data
        data = data.map_partitions(
            coerce_dtypes,
            dataset_schema=dataset_schema,
        )
        data = data.map_partitions(
            Dataset.drop_non_schema_columns,
            dataset_schema=dataset_schema,
        )

        dtype_info = data.map_partitions(get_column_dtypes_info).persist()
        aggregated_dtype_info = aggregate_dtype_info(dtype_info.compute())

        with check:
            assert aggregated_dtype_info["is_expected_dtype"].all()
        object_col_has_wrong_type = (
            aggregated_dtype_info["object_column_types"] != "str"
        ) & (aggregated_dtype_info["object_column_types"] != "")
        with check:
            assert sum(object_col_has_wrong_type) == 0, aggregated_dtype_info.loc[
                object_col_has_wrong_type
            ]

    else:  # pandas
        data_file_paths = get_dataset_filepaths(Path(source), dataset_schema.name)
        filters = get_data_filters(dataset_schema, year, state)
        pandas_data = [
            load_standard_dataset(path, filters, engine) for path in data_file_paths
        ]
        dataset_data = [data for data in pandas_data if len(data) != 0]

        seed = update_seed(SEED, year)
        datasets: list[Dataset] = [
            Dataset(dataset_schema, data, f"{seed}_{i}")  # type: ignore[arg-type]
            for i, data in enumerate(dataset_data)
        ]
        for dataset in datasets:
            clean_input_data(dataset.dataset_schema, dataset.data)
            reformat_dates_for_noising(dataset.dataset_schema, dataset.data)

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
                        prenoised_dataframes,
                        datasets,
                        config,
                        full_dataset_name,
                        fuzzy_checker,
                    )
                    if noise_type.name == NOISE_TYPES.duplicate_with_guardian.name:
                        # noising after duplicate_with_guardian should be done on prenoised data
                        # since it duplicates simulant ID which must be unique to be used as an identifier
                        # TODO: Noise duplicate_with_guardian normally when record IDs
                        # are implemented (MIC-4039)
                        datasets = [
                            Dataset(dataset_schema, data, f"{seed}_{i}")
                            for i, data in enumerate(prenoised_dataframes)
                        ]

            if isinstance(noise_type, ColumnNoiseType):
                for column in dataset_schema.columns:
                    if config.has_noise_type(
                        dataset_schema.name, noise_type.name, column.name
                    ):
                        # don't noise ssa_event_type because it's used as an identifier column
                        # along with simulant id
                        # TODO: Noise ssa_event_type when record IDs are implemented (MIC-4039)
                        if column == COLUMNS.ssa_event_type:
                            continue
                        for dataset in datasets:
                            # noise datasets in place
                            noise_type(dataset, config, column.name)
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
                            column.name,
                            fuzzy_checker,
                        )

        # post-processing tests on final data
        for dataset in datasets:
            # these functions are called by Dataset as part of noising process
            # after noise types have been applied
            dataset.data = coerce_dtypes(dataset.data, dataset.dataset_schema)
            dataset.data = Dataset.drop_non_schema_columns(
                dataset.data, dataset.dataset_schema
            )

            check_column_dtypes(dataset.data)
        # do this outside loop to avoid reading data multiple times
        check_unnoised_id_cols(datasets, dataset.dataset_schema.name)


def check_column_dtypes(
    data: pd.DataFrame,
) -> None:
    """Tests that column dtypes are as expected"""
    for col_name in data.columns:
        col = COLUMNS.get_column(col_name)
        expected_dtype = col.dtype_name
        if expected_dtype == np.dtype(object):
            # str dtype is 'object'
            # Check that they are actually strings and not some other
            # type of object.
            # mypy wants typed type_function to pass into apply but doesn't
            # accept type as an output
            type_function: Callable[..., Any] = lambda x: type(x)
            actual_types = data[col.name].dropna().apply(type_function)
            with check:
                assert (actual_types == str).all(), actual_types.unique()
        with check:
            assert data[col.name].dtype == expected_dtype


def get_column_dtypes_info(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Tests that column dtypes are as expected"""
    dtype_info: dict[str, Any] = {
        "column": [],
        "is_expected_dtype": [],
        "object_column_types": [],
    }
    for col_name in data.columns:
        col = COLUMNS.get_column(col_name)
        expected_dtype = col.dtype_name

        dtype_info["column"].append(col_name)
        dtype_info["is_expected_dtype"].append(data[col.name].dtype == expected_dtype)
        if expected_dtype == np.dtype(object):
            # str dtype is 'object'
            # Check that they are actually strings and not some other
            # type of object.
            # mypy wants typed type_function to pass into apply but doesn't
            # accept type as an output
            type_function: Callable[..., Any] = lambda x: type(x)
            actual_types = [
                type.__name__ for type in data[col.name].dropna().apply(type_function)
            ]
            dtype_info["object_column_types"].append(",".join(set(actual_types)))
        else:
            dtype_info["object_column_types"].append("")

    return pd.DataFrame(dtype_info)


def check_unnoised_id_cols(datasets: list[Dataset], dataset_name: str) -> None:
    """Tests that all datasets retain unnoised simulant_id and household_id
    (except for SSA which does not include household_id)
    """
    unnoised_id_cols = [COLUMNS.simulant_id.name]
    if dataset_name != DATASET_SCHEMAS.ssa.name:
        unnoised_id_cols.append(COLUMNS.household_id.name)
    original = initialize_dataset_with_sample(dataset_name)
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)

    for noised_dataset in datasets:
        check_noised, check_original, _ = get_common_datasets(
            dataset_schema, original.data, noised_dataset.data
        )
        assert (
            (
                check_original.reset_index()[unnoised_id_cols]
                == check_noised.reset_index()[unnoised_id_cols]
            )
            .all()
            .all()
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
    expected_noise_level = 0.0

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
        if noise_type == "copy_from_household_member":
            if column == "age":
                num_sims_with_silent_noising = sum(
                    shared_prenoised.loc[to_compare_idx, column].astype(float)
                    == shared_prenoised.loc[to_compare_idx, f"copy_{column}"].astype(float)
                )
            elif column == "date_of_birth":
                num_sims_with_silent_noising = sum(
                    shared_prenoised.loc[to_compare_idx, column].astype(str)
                    == shared_prenoised.loc[to_compare_idx, f"copy_{column}"].astype(str)
                )
            else:
                num_sims_with_silent_noising = 0
            num_eligible -= num_sims_with_silent_noising

        if noise_type == "swap_month_and_day":
            dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
            date_format = dataset_schema.date_format
            _, month, day = parse_dates(
                shared_prenoised.loc[to_compare_idx, column], date_format
            )
            num_sims_with_same_month_and_day = sum(month == day)
            num_eligible -= num_sims_with_same_month_and_day

        numerator += noise_level
        denominator += num_eligible
        expected_noise_level += expected_noise * num_eligible

    # no eligible rows for unit number in acs data because it's a sparse column and acs is small
    if column == COLUMNS.unit_number.name and denominator == 0:
        return

    fuzzy_checker.fuzzy_assert_proportion(
        name=noise_type,
        observed_numerator=numerator,
        observed_denominator=denominator,
        target_proportion=expected_noise_level / denominator,
        name_additional=f"{dataset_name}_{column}_{noise_type}",
    )


def get_column_noising_counts(
    data: pd.DataFrame,
    config: NoiseConfiguration,
    dataset_name: str,
    noise_type: str,
    column: str,
) -> pd.DataFrame:
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)

    # Validate column noise level
    expected_config_noise = config.get_cell_probability(dataset_name, noise_type, column)
    includes_token_noising = config.has_parameter(
        dataset_name, noise_type, Keys.TOKEN_PROBABILITY, column
    ) or config.has_parameter(
        dataset_name, noise_type, Keys.ZIPCODE_DIGIT_PROBABILITIES, column
    )

    noised_data = get_noised_data(data)
    prenoised_data = get_prenoised_data(data)
    prenoised_data.columns = [col.replace("_prenoised", "") for col in prenoised_data.columns]

    # Check that originally missing data remained missing
    originally_missing_idx = prenoised_data.index[prenoised_data[column].isna()]
    missing_data_not_missing = (
        not noised_data.loc[originally_missing_idx, column].isna().all()
    )

    # Check for noising where applicable
    if len(originally_missing_idx) == len(noised_data):
        return pd.DataFrame(
            {
                "numerator": [0],
                "denominator": [0],
                "expected_numerator": [0],
                "missing_data_not_missing": [missing_data_not_missing],
            }
        )

    # make sure dtypes match when comparing prenoised and noised values
    # adapted from utilities.coerce_dtypes
    if prenoised_data[column].dtype.name != noised_data[column].dtype.name:
        if noised_data[column].dtype.name == DtypeNames.OBJECT:
            prenoised_data[column] = to_string(prenoised_data[column])
        else:
            # mypy doesn't like using a variable as an argument to astype
            prenoised_data[column] = prenoised_data[column].astype(noised_data[column].dtype.name)  # type: ignore [call-overload]

    to_compare_idx = prenoised_data.index.difference(originally_missing_idx)
    rows_noised = (
        prenoised_data.loc[to_compare_idx, column] != noised_data.loc[to_compare_idx, column]
    ).sum()

    if includes_token_noising:
        if noise_type == NOISE_TYPES.write_wrong_zipcode_digits.name:
            token_probability: list[
                float
            ] | int | float = config.get_zipcode_digit_probabilities(dataset_name, column)
        else:
            token_probability = config.get_token_probability(dataset_name, noise_type, column)

        # Get number of tokens per string to calculate expected proportion
        tokens_per_string_getter: Callable[
            ..., pd.Series[int] | int
        ] = TOKENS_PER_STRING_MAPPER.get(noise_type, lambda x: x.astype(str).str.len())
        tokens_per_string: pd.Series[int] | int = tokens_per_string_getter(
            prenoised_data.loc[to_compare_idx, column]
        )
        # Calculate probability no token is noised
        if isinstance(token_probability, list):
            # Calculate write wrong zipcode average digits probability any token is noise
            avg_probability_any_token_noised = 1 - math.prod(
                [1 - p for p in token_probability]
            )
        else:
            # this assert is for mypy
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
    if noise_type == "copy_from_household_member":
        if column == "age":
            num_sims_with_silent_noising = sum(
                prenoised_data.loc[to_compare_idx, column].astype(float)
                == prenoised_data.loc[to_compare_idx, f"copy_{column}"].astype(float)
            )
        elif column == "date_of_birth":
            num_sims_with_silent_noising = sum(
                prenoised_data.loc[to_compare_idx, column].astype(str)
                == prenoised_data.loc[to_compare_idx, f"copy_{column}"].astype(str)
            )
        else:
            num_sims_with_silent_noising = 0
        num_eligible -= num_sims_with_silent_noising

    if noise_type == "swap_month_and_day":
        dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
        date_format = dataset_schema.date_format
        _, month, day = parse_dates(prenoised_data.loc[to_compare_idx, column], date_format)
        num_sims_with_same_month_and_day = sum(month == day)
        num_eligible -= num_sims_with_same_month_and_day

    expected_numerator = expected_noise * num_eligible

    return pd.DataFrame(
        {
            "numerator": [rows_noised],
            "denominator": [num_eligible],
            "expected_numerator": [expected_numerator],
            "missing_data_not_missing": [missing_data_not_missing],
        }
    )
