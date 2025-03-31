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
from tests.utilities import (
    get_single_noise_type_config,
    initialize_dataset_with_sample,
    run_column_noising_tests,
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


@pytest.mark.parametrize("expected_noise", ["default", 0.01])
def test_omit_row(
    expected_noise: str | float,
    unnoised_dataset: Dataset,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that omit_row noising is being applied at the expected proportion."""
    original_data = unnoised_dataset.data
    config_dict = get_single_noise_type_config(dataset_name, NOISE_TYPES.omit_row.name)

    if expected_noise != "default":
        config_dict[dataset_name][Keys.ROW_NOISE][NOISE_TYPES.omit_row.name][
            Keys.ROW_PROBABILITY
        ] = expected_noise
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
    else:
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
        # updating expected_noise from 'default' to actual default value
        expected_noise = config.get_row_probability(dataset_name, NOISE_TYPES.omit_row.name)
    assert isinstance(expected_noise, float)

    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    dataset = Dataset(dataset_schema, original_data, SEED)
    NOISE_TYPES.omit_row(dataset, config)
    noised_data = dataset.data

    fuzzy_checker.fuzzy_assert_proportion(
        name="test_omit_row",
        observed_numerator=len(original_data) - len(noised_data),
        observed_denominator=len(original_data),
        target_proportion=expected_noise,
    )

    assert set(noised_data.columns) == set(original_data.columns)
    assert (noised_data.dtypes == original_data.dtypes).all()


@pytest.mark.parametrize("expected_noise", ["default", 0.03])
def test_do_not_respond(
    expected_noise: str | float,
    unnoised_dataset: Dataset,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that do_not_respond noising is being applied at the expected proportion."""
    # do_not_respond only applies to survey data
    if dataset_name not in [
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.cps.name,
    ]:
        return

    original_data = unnoised_dataset.data
    config_dict = get_single_noise_type_config(dataset_name, NOISE_TYPES.do_not_respond.name)

    if expected_noise != "default":
        config_dict[dataset_name][Keys.ROW_NOISE][NOISE_TYPES.do_not_respond.name][
            Keys.ROW_PROBABILITY
        ] = expected_noise
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
    else:
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
        # updating expected_noise from 'default' to actual default value
        expected_noise = config.get_row_probability(
            dataset_name, NOISE_TYPES.do_not_respond.name
        )
    assert isinstance(expected_noise, float)

    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    dataset = Dataset(dataset_schema, original_data, SEED)
    NOISE_TYPES.do_not_respond(dataset, config)
    noised_data = dataset.data

    # Check ACS and CPS data is scaled properly due to oversampling
    if dataset_name is not DATASET_SCHEMAS.census.name:  # is acs or cps
        expected_noise = 0.5 + expected_noise / 2

    # Test that noising affects expected proportion with expected types
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(original_data) - len(noised_data),
        observed_denominator=len(original_data),
        # 3% uncertainty on either side
        target_proportion=(expected_noise * 0.97, expected_noise * 1.03),
        name_additional=f"noised_data",
    )
    assert set(noised_data.columns) == set(original_data.columns)
    assert (noised_data.dtypes == original_data.dtypes).all()


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


@pytest.mark.parametrize(
    "probabilities",
    [
        {
            "row_probability_in_households_under_18": 1.0,
            "row_probability_in_college_group_quarters_under_24": 1.0,
        },
        {
            "row_probability_in_households_under_18": 0.7,
            "row_probability_in_college_group_quarters_under_24": 0.8,
        },
    ],
)
def test_guardian_duplication(
    dataset_params: tuple[
        str,
        Callable[..., pd.DataFrame],
        str | None,
        int | None,
        str | None,
        Literal["pandas", "dask"],
    ],
    dataset_name: str,
    probabilities: dict[str, float],
    fuzzy_checker: FuzzyChecker,
    mocker: MockerFixture,
) -> None:
    if dataset_name != DatasetNames.CENSUS:
        return

    # patch these to avoid updating dtypes and dropping columns we need for testing
    mocker.patch("pseudopeople.dataset.coerce_dtypes", side_effect=lambda df, _: df)
    mocker.patch(
        "pseudopeople.dataset.Dataset.drop_non_schema_columns", side_effect=lambda df, _: df
    )
    # allow all irrelevant probabilities to be 0 in our config
    mocker.patch(
        "pseudopeople.configuration.generator.validate_overrides",
        side_effect=lambda *args: None,
    )
    # allow our noise levels to be high in testing
    mocker.patch(
        "pseudopeople.configuration.generator.validate_noise_level_proportions",
        lambda *args: None,
    )

    # get unnoised data
    _, _, source, year, state, engine = dataset_params
    unnoised = generate_decennial_census(
        source=source, config=NO_NOISE, year=year, state=state, engine=engine
    )

    # get noised data using custom config
    config_dict = get_single_noise_type_config(
        dataset_name, NOISE_TYPES.duplicate_with_guardian.name
    )
    for probability_key in [
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24,
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18,
    ]:
        config_dict[dataset_name][Keys.ROW_NOISE][NOISE_TYPES.duplicate_with_guardian.name][
            probability_key
        ] = probabilities[probability_key]
    config = NoiseConfiguration(LayeredConfigTree(config_dict))
    noised = generate_decennial_census(
        source=source, config=config.to_dict(), year=year, state=state, engine=engine
    )

    duplicated = noised.loc[noised["simulant_id"].duplicated()]
    duplicated["age"] = duplicated["age"].astype(int)

    # add old housing type data to duplicated simulants
    old_housing_data = unnoised[["simulant_id", "housing_type"]].rename(
        {"housing_type": "unnoised_housing_type"}, axis=1
    )
    duplicated = duplicated.merge(old_housing_data)

    # separate tests for household under 18 and for college under 24
    for probability_name, age, housing_type in zip(
        [
            Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18,
            Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24,
        ],
        [18, 24],
        ["Household", "College"],
    ):
        group_data = unnoised.loc[
            (unnoised["age"].astype(int) < age)
            & (unnoised["housing_type"] == housing_type)
            & (unnoised["guardian_1"].notna())
        ]
        merged_data = merge_dependents_and_guardians(
            group_data, unnoised
        )  # type:  ignore [arg-type]
        sims_eligible_for_duplication = merged_data.index[
            (
                (merged_data["household_id"] != merged_data["guardian_1_household_id"])
                & (merged_data["guardian_1_household_id"].notna())
            )
            | (
                (merged_data["household_id"] != merged_data["guardian_2_household_id"])
                & (merged_data["guardian_2_household_id"].notna())
            )
        ]
        duplicated_in_group = duplicated.loc[
            (duplicated["age"] < age) & (duplicated["unnoised_housing_type"] == housing_type)
        ]

        fuzzy_checker.fuzzy_assert_proportion(
            name="test_duplicate_guardian",
            observed_numerator=len(duplicated_in_group),
            observed_denominator=len(sims_eligible_for_duplication),
            target_proportion=probabilities[probability_name],
            name_additional=f"noised_data",
        )
    # Only duplicate a dependent one time
    assert noised["simulant_id"].value_counts().max() == 2

    # Check address information is copied in new rows
    guardians = unnoised.loc[
        unnoised["simulant_id"].isin(unnoised["guardian_1"])
        | unnoised["simulant_id"].isin(unnoised["guardian_2"])
    ]
    simulant_ids = unnoised["simulant_id"].values

    for i in duplicated.index:
        dependent = duplicated.loc[i]

        for column in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS:
            guardian_1 = dependent["guardian_1"]
            guardian_2 = dependent["guardian_2"]

            if guardian_2 is np.nan:
                guardians_values = [
                    guardians.loc[guardians["simulant_id"] == guardian_1, column].values[0]
                ]
            else:  # dependent has both guardians
                guardians_values = []
                for guardian in [guardian_1, guardian_2]:
                    if (
                        guardian in simulant_ids
                    ):  # duplicates will not have addresses copied from guardians not in data
                        guardians_values += [
                            guardians.loc[
                                guardians["simulant_id"] == guardian, column
                            ].values[0]
                        ]

            assert dependent[column] in guardians_values


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
