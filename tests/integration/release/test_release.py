from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.entities import NO_NOISE
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.dataset import Dataset
from pseudopeople.interface import generate_decennial_census
from pseudopeople.noise_entities import NOISE_TYPES
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
        expected_noise: float = config.get_row_probability(
            dataset_name, NOISE_TYPES.omit_row.name
        )
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
        expected_noise: float = config.get_row_probability(
            dataset_name, NOISE_TYPES.do_not_respond.name
        )

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
        target_proportion=expected_noise,
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


@pytest.mark.parametrize("duplication_probability", [1.0, 0.8])
def test_guardian_duplication(dataset_params, dataset_name: str, duplication_probability: float, fuzzy_checker: FuzzyChecker, mocker: MockerFixture) -> None:
    if dataset_name != DatasetNames.CENSUS:
        return

    def _merge_dependents_and_guardians(
        dependents_df: pd.DataFrame, full_data: pd.DataFrame
    ) -> pd.DataFrame:
        # Merge dependents with their guardians. We have to merge twice to check
        # if either guardian is living at a separate location from the dependent.
        guardian_1s = full_data.loc[
            full_data["simulant_id"].isin(full_data["guardian_1"]),
            GUARDIAN_DUPLICATION_ADDRESS_COLUMNS + ["simulant_id"],
        ].add_prefix("guardian_1_")
        dependents_and_guardians_df = dependents_df.merge(
            guardian_1s,
            how="left",
            left_on=["guardian_1", "year"],
            right_on=["guardian_1_simulant_id", "guardian_1_year"],
        )
        del guardian_1s
        guardian_2s = full_data.loc[
            full_data["simulant_id"].isin(full_data["guardian_2"]),
            GUARDIAN_DUPLICATION_ADDRESS_COLUMNS + ["simulant_id"],
        ].add_prefix("guardian_2_")
        dependents_and_guardians_df = dependents_and_guardians_df.merge(
            guardian_2s,
            how="left",
            left_on=["guardian_2", "year"],
            right_on=["guardian_2_simulant_id", "guardian_2_year"],
        )
        del guardian_2s

        return dependents_and_guardians_df

    mocker.patch("pseudopeople.dataset.coerce_dtypes", side_effect=lambda df, _: df)
    mocker.patch("pseudopeople.dataset.Dataset.keep_schema_columns", side_effect=lambda df, _: df)
    mocker.patch("pseudopeople.configuration.generator.validate_overrides", side_effect=lambda *args: None)

    # get unnoised data
    _, _, source, year, state, engine = dataset_params
    unnoised = generate_decennial_census(source=source, config=NO_NOISE, year=year, state=state, engine=engine)

    # get noised data using custom config
    config_dict = get_single_noise_type_config(dataset_name, NOISE_TYPES.duplicate_with_guardian.name)
    for probability_key in [Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24, Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18]:
        config_dict[dataset_name][Keys.ROW_NOISE][NOISE_TYPES.duplicate_with_guardian.name][
            probability_key
        ] = duplication_probability
    config = NoiseConfiguration(LayeredConfigTree(config_dict))
    noised = generate_decennial_census(source=source, config=config.to_dict(), year=year, state=state, engine=engine)
    
    duplicated = noised.loc[noised["simulant_id"].duplicated()]
    duplicated['age'] = duplicated['age'].astype(int)

    in_households_under_18 = unnoised.loc[
        (unnoised["age"].astype(int) < 18)
        & (unnoised["housing_type"] == "Household")
        & (unnoised["guardian_1"].notna())
    ]
    in_college_under_24 = unnoised.loc[
        (unnoised["age"].astype(int) < 24)
        & (unnoised["housing_type"] == "College")
        & (unnoised["guardian_1"].notna())
    ]

    merged_18 = _merge_dependents_and_guardians(in_households_under_18, unnoised)
    sims_18_eligible_for_duplication = merged_18.index[
            ((merged_18["household_id"] != merged_18["guardian_1_household_id"])
            & (merged_18["guardian_1_household_id"].notna()))
            | ((merged_18["household_id"] != merged_18["guardian_2_household_id"])
            & (merged_18["guardian_2_household_id"].notna()))
        ]

    merged_24 = _merge_dependents_and_guardians(in_college_under_24, unnoised)
    sims_24_eligible_for_duplication = merged_24.index[
            ((merged_24["household_id"] != merged_24["guardian_1_household_id"])
            & (merged_24["guardian_1_household_id"].notna()))
            | ((merged_24["household_id"] != merged_24["guardian_2_household_id"])
            & (merged_24["guardian_2_household_id"].notna()))
        ]

    duplicated_in_households_under_18 = duplicated.query("age < 18 and old_housing_type=='Household'")
    duplicated_in_college_under_24 = duplicated.query("age < 24 and old_housing_type=='College'")
    
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_duplicate_guardian",
        observed_numerator=len(duplicated_in_households_under_18),
        observed_denominator=len(sims_18_eligible_for_duplication),
        target_proportion=duplication_probability,
        name_additional=f"noised_data",
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_duplicate_guardian",
        observed_numerator=len(duplicated_in_college_under_24),
        observed_denominator=len(sims_24_eligible_for_duplication),
        target_proportion=duplication_probability,
        name_additional=f"noised_data",
    )
    # Only duplicate a dependent one time
    assert noised["simulant_id"].value_counts().max() == 2

    # Check address information is copied in new rows
    guardians = unnoised.loc[unnoised["simulant_id"].isin(unnoised["guardian_1"]) | unnoised["simulant_id"].isin(unnoised["guardian_2"])]
    simulant_ids = unnoised['simulant_id'].values
    guardian_1_ids = [x for x in unnoised['guardian_1'].unique() if not np.isnan(float(x))]
    guardian_2_ids = [x for x in unnoised['guardian_2'].unique() if not np.isnan(float(x))]
    missing_1_ids = [x for x in guardian_1_ids if x not in simulant_ids]
    missing_2_ids = [x for x in guardian_2_ids if x not in simulant_ids]

    for i in duplicated.index:
        dependent = duplicated.loc[i]
        for column in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS:
            guardian_1 = dependent["guardian_1"]
            guardian_2 = dependent["guardian_2"]
            #if guardian_1 in missing_1_ids or guardian_2 in missing_2_ids:
            #    pass
            if False:
               pass
            else:
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
