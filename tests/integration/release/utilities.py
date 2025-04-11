from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd
from pytest_check import check
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_functions import merge_dependents_and_guardians
from pseudopeople.schema_entities import DATASET_SCHEMAS


def run_omit_row_tests(
    original_data: pd.DataFrame | dd.DataFrame,
    noised_data: pd.DataFrame | dd.DataFrame,
    config: NoiseConfiguration,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that omit_row noising is being applied at the expected proportion."""
    expected_noise = config.get_row_probability(dataset_name, NOISE_TYPES.omit_row.name)
    with check:
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_omit_row",
            observed_numerator=len(original_data) - len(noised_data),
            observed_denominator=len(original_data),
            target_proportion=expected_noise,
        )

    with check:
        assert set(noised_data.columns) == set(original_data.columns)
    with check:
        assert (noised_data.dtypes == original_data.dtypes).all()


def run_do_not_respond_tests(
    original_data: pd.DataFrame | dd.DataFrame,
    noised_data: pd.DataFrame | dd.DataFrame,
    config: NoiseConfiguration,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that do_not_respond noising is being applied at the expected proportion."""
    expected_noise = config.get_row_probability(dataset_name, NOISE_TYPES.do_not_respond.name)

    # Check ACS and CPS data is scaled properly due to oversampling
    if dataset_name is not DATASET_SCHEMAS.census.name:  # is acs or cps
        expected_noise = 0.5 + expected_noise / 2

    # Test that noising affects expected proportion with expected types
    with check:
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_do_not_respond",
            observed_numerator=len(original_data) - len(noised_data),
            observed_denominator=len(original_data),
            # 3% uncertainty on either side
            target_proportion=(expected_noise * 0.97, expected_noise * 1.03),
            name_additional=f"noised_data",
        )
    with check:
        assert set(noised_data.columns) == set(original_data.columns)
    with check:
        assert (noised_data.dtypes == original_data.dtypes).all()


def run_guardian_duplication_tests(
    original_data: pd.DataFrame | dd.DataFrame,
    noised_data: pd.DataFrame | dd.DataFrame,
    config: NoiseConfiguration,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    # get duplicated simulants
    duplicated = noised_data.loc[noised_data["simulant_id"].duplicated()]
    duplicated["age"] = duplicated["age"].astype(int)

    # add old housing type data to duplicated simulants
    old_housing_data = original_data[["simulant_id", "housing_type"]].rename(
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
        group_data = original_data.loc[
            (original_data["age"].astype(int) < age)
            & (original_data["housing_type"] == housing_type)
            & (original_data["guardian_1"].notna())
        ]
        merged_data = merge_dependents_and_guardians(
            group_data, original_data
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

        expected_noise = config.get_duplicate_with_guardian_probabilities(
            DATASET_SCHEMAS.census.name, probability_name
        )
        with check:
            fuzzy_checker.fuzzy_assert_proportion(
                name="test_duplicate_guardian",
                observed_numerator=len(duplicated_in_group),
                observed_denominator=len(sims_eligible_for_duplication),
                target_proportion=expected_noise,
                name_additional=f"noised_data",
            )

    # Only duplicate a dependent one time
    with check:
        assert noised_data["simulant_id"].value_counts().max() == 2

    # Check address information is copied in new rows
    guardians = original_data.loc[
        original_data["simulant_id"].isin(original_data["guardian_1"])
        | original_data["simulant_id"].isin(original_data["guardian_2"])
    ]
    simulant_ids = original_data["simulant_id"].values

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
            with check:
                assert dependent[column] in guardians_values
