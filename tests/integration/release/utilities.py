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
from pseudopeople.dataset import Dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_functions import merge_dependents_and_guardians
from pseudopeople.schema_entities import DATASET_SCHEMAS


def run_do_not_respond_tests(
    prenoised_dataframes: list[pd.DataFrame],
    noised_datasets: list[Dataset],
    config: NoiseConfiguration,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Takes a list of prenoised DataFrames and matching noised Datasets and checks that
    after applying do_not_respond, 1) columns and dtypes were preserved properly and
    2) the expected number of rows were removed."""
    expected_noise = config.get_row_probability(dataset_name, NOISE_TYPES.do_not_respond.name)

    # ACS and CPS data are oversampled by a factor of 2 so we apply a baseline probability of
    # 0.5 for do_not_respond so the data returned is of the expected size
    # # TODO: [MIC-6002] move this processing step out of noising
    if dataset_name is not DATASET_SCHEMAS.census.name:  # is acs or cps
        expected_noise = 0.5 + expected_noise / 2

    numerator = 0
    denominator = 0
    for prenoised_dataframe, noised_dataset in zip(prenoised_dataframes, noised_datasets):
        # Do all the calculations and non-fuzzy checks
        with check:
            assert set(noised_dataset.data.columns) == set(prenoised_dataframe.columns)
        with check:
            assert (noised_dataset.data.dtypes == prenoised_dataframe.dtypes).all()

        shard_numerator = len(prenoised_dataframe) - len(noised_dataset.data)
        shard_denominator = len(prenoised_dataframe)

        numerator += shard_numerator
        denominator += shard_denominator

    # Test that noising affects expected proportion with expected types
    with check:
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_do_not_respond",
            observed_numerator=numerator,
            observed_denominator=denominator,
            # 3% uncertainty on either side
            target_proportion=(expected_noise * 0.97, expected_noise * 1.03),
            name_additional=f"noised_data",
        )


def run_omit_row_tests(
    prenoised_dataframes: list[pd.DataFrame],
    noised_datasets: list[Dataset],
    config: NoiseConfiguration,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Takes a list of prenoised DataFrames and matching noised Datasets and checks that
    after applying omit_row, 1) columns and dtypes were preserved properly and
    2) the expected number of rows were removed."""
    expected_noise = config.get_row_probability(dataset_name, NOISE_TYPES.omit_row.name)

    numerator = 0
    denominator = 0

    for prenoised_dataframe, noised_dataset in zip(prenoised_dataframes, noised_datasets):
        # Do all the calculations and non-fuzzy checks
        with check:
            assert set(noised_dataset.data.columns) == set(prenoised_dataframe.columns)
        with check:
            assert (noised_dataset.data.dtypes == prenoised_dataframe.dtypes).all()

        shard_numerator = len(prenoised_dataframe) - len(noised_dataset.data)
        shard_denominator = len(prenoised_dataframe)

        numerator += shard_numerator
        denominator += shard_denominator

    # Test that noising affects expected proportion with expected types
    with check:
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_omit_row",
            observed_numerator=numerator,
            observed_denominator=denominator,
            # 3% uncertainty on either side
            target_proportion=(expected_noise * 0.97, expected_noise * 1.03),
            name_additional=f"noised_data",
        )


def run_guardian_duplication_tests(
    prenoised_dataframes: list[pd.DataFrame],
    noised_datasets: list[Dataset],
    config: NoiseConfiguration,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Takes a list of prenoised DataFrames and matching noised Datasets and checks that
    after applying duplicate_with_guardian, 1) no simulants are duplicated more than once,
    2) address information was duplicated correctly and 3) the expected number of simulants
    were duplicated."""
    numerators = {
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0,
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0,
    }
    denominators = {
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0,
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0,
    }

    for prenoised_dataframe, noised_dataset in zip(prenoised_dataframes, noised_datasets):
        # get duplicated simulants
        duplicated = noised_dataset.data.loc[noised_dataset.data["simulant_id"].duplicated()]

        # add old housing type data to duplicated simulants
        old_housing_data = prenoised_dataframe[["simulant_id", "housing_type"]].rename(
            {"housing_type": "unnoised_housing_type"}, axis=1
        )
        duplicated = duplicated.merge(old_housing_data)

        # collect data for fuzzy checking for household under 18 and for college under 24
        for probability_name, max_age, housing_type in zip(
            [
                Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18,
                Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24,
            ],
            [18, 24],
            ["Household", "College"],
        ):
            group_data = prenoised_dataframe.loc[
                (prenoised_dataframe["age"] < max_age)
                & (prenoised_dataframe["housing_type"] == housing_type)
                & (prenoised_dataframe["guardian_1"].notna())
            ]
            merged_data = merge_dependents_and_guardians(group_data, prenoised_dataframe)
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
                (duplicated["age"] < max_age)
                & (duplicated["unnoised_housing_type"] == housing_type)
            ]

            numerators[probability_name] += len(duplicated_in_group)
            denominators[probability_name] += len(sims_eligible_for_duplication)

        # Only duplicate a dependent one time
        with check:
            assert noised_dataset.data["simulant_id"].value_counts().max() == 2

        # Check address information is copied in new rows
        guardians = prenoised_dataframe.loc[
            prenoised_dataframe["simulant_id"].isin(prenoised_dataframe["guardian_1"])
            | prenoised_dataframe["simulant_id"].isin(prenoised_dataframe["guardian_2"])
        ]
        simulant_ids = prenoised_dataframe["simulant_id"].values

        for i in duplicated.index:
            dependent = duplicated.loc[i]

            for column in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS:
                guardian_1 = dependent["guardian_1"]
                guardian_2 = dependent["guardian_2"]

                if guardian_2 is np.nan:
                    guardians_values = [
                        guardians.loc[guardians["simulant_id"] == guardian_1, column].values[
                            0
                        ]
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

    # fuzzy checks for both groups
    for probability_name in [
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18,
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24,
    ]:
        expected_noise = config.get_duplicate_with_guardian_probabilities(
            DATASET_SCHEMAS.census.name, probability_name
        )

        with check:
            fuzzy_checker.fuzzy_assert_proportion(
                name="test_duplicate_guardian",
                observed_numerator=numerators[probability_name],
                observed_denominator=denominators[probability_name],
                target_proportion=expected_noise,
                name_additional=f"noised_data",
            )
