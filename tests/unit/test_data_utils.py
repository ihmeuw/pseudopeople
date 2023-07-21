import pandas as pd
import pytest

from pseudopeople.loader import combine_joint_filers, flatten_data
from pseudopeople.schema_entities import COLUMNS


@pytest.fixture(scope="module")
def dummy_1040():
    return pd.DataFrame(
        {
            COLUMNS.simulant_id.name: list(range(5)) * 2,
            COLUMNS.joint_filer.name: [
                False,
                True,
                False,
                True,
                False,
            ]
            * 2,
            COLUMNS.relation_to_reference_person.name: [
                "Reference person",
                "Opp-sex spouse",
                "Reference person",
                "Opp-sex spouse",
                "Reference person",
            ]
            * 2,
            COLUMNS.household_id.name: [
                10,
                10,
                11,
                11,
                12,
            ]
            * 2,
            COLUMNS.tax_year.name: [2020] * 5 + [2021] * 5,
        }
    )


@pytest.fixture(scope="module")
def dummy_tax_dependents():
    return pd.DataFrame(
        {
            COLUMNS.simulant_id.name: list(range(100, 105)) * 2,
            COLUMNS.guardian_id.name: [0, 0, 0, 1, 2] * 2,
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
            ],
            COLUMNS.tax_year.name: [2020] * 5 + [2021] * 5,
        }
    )


def test_combine_joint_filers(dummy_1040):
    joint_1040 = combine_joint_filers(dummy_1040)

    assert set(joint_1040.columns) == set(
        [
            COLUMNS.simulant_id.name,
            COLUMNS.relation_to_reference_person.name,
            COLUMNS.joint_filer.name,
            COLUMNS.household_id.name,
            COLUMNS.tax_year.name,
            COLUMNS.spouse_simulant_id.name,
            COLUMNS.spouse_relation_to_reference_person.name,
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
    # Guardian/simulant id 0 has 3 dependents which is the highest number of dependents
    # Make sure we do not have extra columns
    # fix me: find way to improve this
    assert "dependent_4_favorite_food" not in dependents_wide.columns
