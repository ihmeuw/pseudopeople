import numpy as np
import pandas as pd
import pytest

from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.loader import (
    combine_joint_filers,
    combine_ssn_and_itin_columns,
    flatten_data,
    load_and_prep_1040_data,
)
from pseudopeople.schema_entities import COLUMNS, DATASETS


@pytest.fixture(scope="module")
def dummy_1040():
    return pd.DataFrame(
        {
            COLUMNS.simulant_id.name: list(range(8)) * 2,
            COLUMNS.joint_filer.name: [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
            ]
            * 2,
            COLUMNS.relationship_to_reference_person.name: [
                "Reference person",
                "Opp-sex spouse",
                "Biological child",
                "Reference person",
                "Opp-sex spouse",
                "Reference person",
                "Roommate",
                "Opp-sex spouse",
            ]
            * 2,
            COLUMNS.household_id.name: [
                10,
                10,
                10,
                11,
                11,
                12,
                13,
                14,
            ]
            * 2,
            COLUMNS.tax_year.name: [2020] * 8 + [2021] * 8,
        }
    )


@pytest.fixture(scope="module")
def dummy_tax_dependents():
    return pd.DataFrame(
        {
            COLUMNS.simulant_id.name: [2, 103, 104, 105, 106, 107, 108, 109] * 2,
            COLUMNS.guardian_id.name: [0, 0, 0, 0, 0, 3, 3, 5] * 2,
            "favorite_food": [
                "Pizza",
                "Cookie",
                "Ice cream",
                "Cheeseburger",
                "Sandwich",
                "Salad",
                "Tacos",
                "Pasta",
                "Ramen",
                "Waffles",
                "Cookies",
                "Watermelon",
                "Nachos",
                "BBQ",
                "Bagel",
                "Grapes",
            ],
            COLUMNS.tax_year.name: [2020] * 8 + [2021] * 8,
        }
    )


def test_combine_joint_filers(dummy_1040):
    joint_1040 = combine_joint_filers(dummy_1040)

    assert set(joint_1040.columns) == set(
        [
            COLUMNS.simulant_id.name,
            COLUMNS.relationship_to_reference_person.name,
            COLUMNS.joint_filer.name,
            COLUMNS.household_id.name,
            COLUMNS.tax_year.name,
            COLUMNS.spouse_simulant_id.name,
            COLUMNS.spouse_relationship_to_reference_person.name,
            COLUMNS.spouse_joint_filer.name,
            COLUMNS.spouse_tax_year.name,
            COLUMNS.spouse_household_id.name,
        ]
    )
    joint_filer_ids = dummy_1040.loc[
        dummy_1040[COLUMNS.joint_filer.name] == True, COLUMNS.simulant_id.name
    ]
    # Joint filer ids should not be in simulant ids
    assert not bool(set(joint_1040[COLUMNS.simulant_id.name]) & set(joint_filer_ids))
    # Check we are returned correct number of rows... original data - joint filer rows
    assert len(dummy_1040) - len(joint_filer_ids) == len(joint_1040)


def test_flatten_data(dummy_tax_dependents):
    dependents_wide = flatten_data(
        data=dummy_tax_dependents,
        index_cols=[COLUMNS.guardian_id.name, COLUMNS.tax_year.name],
        rank_col=COLUMNS.simulant_id.name,
        value_cols=["favorite_food"],
    )
    # Dependent and guardian ids should never overlap
    assert not bool(
        set(dependents_wide.reset_index()[COLUMNS.guardian_id.name])
        & set(dummy_tax_dependents[COLUMNS.simulant_id.name])
    )
    # The length of rows should be total guardian/tax year combinations which is 6
    assert len(dependents_wide) == 6
    # Guardian/simulant id 0 has 4 dependents which is the highest number of dependents
    # Make sure we do not have extra columns - more than 4 dependent. When only have one
    # "value" column from our pivot so we can assert there should be 4 columns for 4 dependents.
    assert len(dependents_wide.columns) == 4
    # Assert expected nans for depdents 2, 3, 4 columns - we have 3 guardians (0, 3, 5) with
    # 5, 2, and 1 dependents respectively. We expected dependent 2 column to have 2 nans, dependent
    # 3 and dependent 4 columns to have 4 nans.
    assert dependents_wide["2_favorite_food"].isna().sum() == 2
    for dependent in ["3", "4"]:
        assert dependents_wide[f"{dependent}_favorite_food"].isna().sum() == 4


def test_load_and_prep_1040_data():
    tax_dataset_names = [
        DatasetNames.TAXES_1040,
        # DatasetNames.TAXES_W2_1099,
        DatasetNames.TAXES_DEPENDENTS,
    ]
    tax_dataset_filepaths = {
        tax_dataset: paths.SAMPLE_DATA_ROOT / tax_dataset / f"{tax_dataset}.parquet"
        for tax_dataset in tax_dataset_names
    }
    tax_1040 = load_and_prep_1040_data(tax_dataset_filepaths, user_filters=[])

    # No joint filer should be in the formatted simulant_id column
    # We must check each year because of migration/joint filing
    for year in tax_1040[COLUMNS.tax_year.name].unique():
        year_df = tax_1040.loc[tax_1040[COLUMNS.tax_year.name] == year]
        assert not bool(
            set(year_df[COLUMNS.simulant_id.name])
            & set(year_df[COLUMNS.spouse_simulant_id.name])
        )
    # Check formatted tax 1040 has necessary output columns
    # Note this is before we clense our data of extra columns
    tax_1040_dataset_cols = [column.name for column in DATASETS.tax_1040.columns]
    assert set(tax_1040_dataset_cols).issubset(set(tax_1040.columns))


def test_combine_ssn_itin_column_combine():
    # The purpose of this function is to test the logic merging the two columns
    # of ssn and itin is correct using np.where
    df = pd.DataFrame(
        {
            COLUMNS.ssn.name: [
                "123-45-6789",
                np.nan,
                "987-65-4321",
                np.nan,
                "543-67-2189",
            ],
            COLUMNS.itin.name: [
                np.nan,
                "900-10-5555",
                np.nan,
                "111-222-3333",
                np.nan,
            ],
        }
    )
    # Get mask for nulls in ssn
    swap_mask = df[COLUMNS.ssn.name].isna()
    df = combine_ssn_and_itin_columns(df)

    # There should be no nans in ssn column
    assert df[COLUMNS.ssn.name].isna().sum() == 0
    # New ssn values should be old itin values
    assert (df[COLUMNS.ssn.name][swap_mask] == df[COLUMNS.itin.name][swap_mask]).all()
