from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.dataset import Dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from tests.integration.conftest import IDX_COLS, _get_common_datasets, get_unnoised_data
from tests.utilities import (
    initialize_dataset_with_sample,
    run_column_noising_tests,
    run_omit_row_or_do_not_respond_tests,
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


def test_row_noising_omit_row_or_do_not_respond(
    noised_data: pd.DataFrame,
    dataset_name: str,
    config: dict[str, Any],
) -> None:
    """Tests that omit_row and do_not_respond row noising are being applied"""
    idx_cols = IDX_COLS.get(dataset_name)
    original = get_unnoised_data(dataset_name)
    original_data = original.data.set_index(idx_cols)
    noised_data = noised_data.set_index(idx_cols)

    run_omit_row_or_do_not_respond_tests(dataset_name, config, original_data, noised_data)


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
            actual_types = noised_data[col.name].dropna().apply(type)
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


@pytest.mark.parametrize("duplication_probability", [0.8])
def test_guardian_duplication(unnoised_dataset: Dataset, dataset_name: str, duplication_probability: float, fuzzy_checker: FuzzyChecker) -> None:
    if dataset_name != DatasetNames.CENSUS:
        return
    dummy_data = unnoised_dataset.data
    
    # config: NoiseConfiguration = get_configuration()
    # config._update(
    #     {
    #         DATASET_SCHEMAS.census.name: {
    #             Keys.ROW_NOISE: {
    #                 NOISE_TYPES.duplicate_with_guardian.name: {
    #                     Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: duplication_probability,
    #                     Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: duplication_probability,
    #                 },
    #             },
    #         }
    #     }
    # )
    config_dict = get_single_noise_type_config(DatasetNames.CENSUS, "duplicate_with_guardian")
    config = NoiseConfiguration(LayeredConfigTree(config_dict))
    breakpoint()
    census = Dataset(DATASET_SCHEMAS.census, dummy_data, 0)
    NOISE_TYPES.duplicate_with_guardian(census, config)
    noised = census.data

    duplicated = noised.loc[noised["simulant_id"].duplicated()]

    if duplication_probability == 1.0:
        # We know the following since every dependent is duplicated
        # in the case of 100% duplication:
        #  - Simulant ids 0, 1, 2, 3, and-5 will all be duplicated
        #  - Simulant ids 5-9 are guardians. The only overlap is simulant id 5,
        #    who is both a dependent and a guardian
        #  - Simulant id 0 and 3 have two guardians, 8 and 9.

        # Check that the correct rows were duplicated. Duplicated returns all
        # instances of True after the first instance
        assert len(noised) == len(dummy_data) + len(duplicated)
        assert set(duplicated["simulant_id"].tolist()) == set(["0", "1", "2", "3", "5"])
    else: # non-1 probability
        has_guardian = dummy_data['guardian_1'].notna()
        not_in_military = dummy_data['housing_type'] != 'Military'
        num_eligible_for_duplication = sum(has_guardian & not_in_military)

        fuzzy_checker.fuzzy_assert_proportion(
            name="test_do_not_respond",
            observed_numerator=len(duplicated),
            observed_denominator=num_eligible_for_duplication,
            target_proportion=duplication_probability,
            name_additional=f"noised_data",
        )
    # Only duplicate a dependent one time
    assert noised["simulant_id"].value_counts().max() == 2

    # Check address information is copied in new rows
    guardians = dummy_data.loc[dummy_data["simulant_id"].isin(dummy_data["guardian_1"])]
    for i in duplicated.index:
        dependent = duplicated.loc[i]
        for column in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS:
            guardian_1 = dependent["guardian_1"]
            guardian_2 = dependent["guardian_2"]
            if guardian_2 is np.nan:
                guardians_values = [
                    guardians.loc[guardians["simulant_id"] == guardian_1, column].values[0]
                ]
            else:
                guardians_values = [
                    guardians.loc[guardians["simulant_id"] == guardian_1, column].values[0],
                    guardians.loc[guardians["simulant_id"] == guardian_2, column].values[0],
                ]
            assert dependent[column] in guardians_values