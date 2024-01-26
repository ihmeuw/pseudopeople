import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_functions import _get_census_omission_noise_levels
from pseudopeople.schema_entities import DATASETS
from tests.conftest import FuzzyChecker

RANDOMNESS = RandomnessStream(
    key="test_row_noise",
    clock=lambda: pd.Timestamp("2020-09-01"),
    seed=0,
    index_map=IndexMap(),
)


@pytest.fixture()
def dummy_data():
    num_simulants = 1_000_000
    dummy_idx = pd.Index(range(num_simulants))

    numbers_list = list(range(10))
    numbers_series = pd.Series(numbers_list * int(num_simulants / len(numbers_list)))
    word_list = ["word_1", "word_2", "word_3", "word_4", "word5"]
    word_series = pd.Series(word_list * int(num_simulants / len(word_list)))

    return pd.DataFrame(
        {
            "numbers": numbers_series,
            "words": word_series,
        },
        index=dummy_idx,
    )


def test_omit_row(dummy_data, fuzzy_checker: FuzzyChecker):
    config = get_configuration()[DATASETS.tax_w2_1099.name][Keys.ROW_NOISE][
        NOISE_TYPES.omit_row.name
    ]
    dataset_name_1 = "dummy_dataset_name"
    noised_data1 = NOISE_TYPES.omit_row(dataset_name_1, dummy_data, config, RANDOMNESS)

    expected_noise_1 = config[Keys.ROW_PROBABILITY]
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_omit_row",
        observed_numerator=len(noised_data1),
        observed_denominator=len(dummy_data),
        target_proportion=1 - expected_noise_1,
    )
    assert set(noised_data1.columns) == set(dummy_data.columns)
    assert (noised_data1.dtypes == dummy_data.dtypes).all()


def test_do_not_respond(mocker, dummy_data, fuzzy_checker: FuzzyChecker):
    config = get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.do_not_respond.name
    ]
    mocker.patch(
        "pseudopeople.noise_functions._get_census_omission_noise_levels",
        side_effect=(lambda *_: config[Keys.ROW_PROBABILITY]),
    )
    dataset_name_1 = DATASETS.census.name
    dataset_name_2 = DATASETS.acs.name
    my_dummy_data = dummy_data.copy()
    my_dummy_data["age"] = 27
    my_dummy_data["sex"] = "Female"
    my_dummy_data["race_ethnicity"] = "Vulcan"
    noised_data1 = NOISE_TYPES.do_not_respond(
        dataset_name_1, my_dummy_data, config, RANDOMNESS
    )
    noised_data2 = NOISE_TYPES.do_not_respond(
        dataset_name_2, my_dummy_data, config, RANDOMNESS
    )

    # Test that noising affects expected proportion with expected types
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(my_dummy_data) - len(noised_data1),
        observed_denominator=len(my_dummy_data),
        target_proportion=config[Keys.ROW_PROBABILITY],
        name_additional=f"noised_data1",
    )
    assert set(noised_data1.columns) == set(my_dummy_data.columns)
    assert (noised_data1.dtypes == my_dummy_data.dtypes).all()

    # Check ACS data is scaled properly due to oversampling
    expected_noise = 0.5 + config[Keys.ROW_PROBABILITY] / 2
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(my_dummy_data) - len(noised_data2),
        observed_denominator=len(my_dummy_data),
        target_proportion=expected_noise,
        name_additional=f"noised_data2",
    )
    assert set(noised_data2.columns) == set(my_dummy_data.columns)
    assert (noised_data2.dtypes == my_dummy_data.dtypes).all()
    assert len(noised_data1) != len(noised_data2)
    assert True


@pytest.mark.parametrize(
    "age, race_ethnicity, sex, expected_level",
    [
        (3, "White", "Female", 0.0091),
        (35, "Black", "Male", 0.0611),
        (55, "Asian", "Female", 0),
    ],
)
def test__get_census_omission_noise_levels(age, race_ethnicity, sex, expected_level):
    """Test helper function for do_not_respond noising based on demography of age, race/ethnicity, and sex"""
    pop = pd.DataFrame(
        [[age, race_ethnicity, sex] for i in range(10)],
        index=range(10),
        columns=["age", "race_ethnicity", "sex"],
    )
    result = _get_census_omission_noise_levels(pop)
    assert (np.isclose(result, expected_level, rtol=0.0001)).all()


def test_do_not_respond_missing_columns(dummy_data):
    """Test do_not_respond only applies to expected datasets."""
    config = get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.do_not_respond.name
    ]
    with pytest.raises(ValueError, match="missing required columns"):
        _ = NOISE_TYPES.do_not_respond("silly_dataset", dummy_data, config, RANDOMNESS)


def test_guardian_duplication():
    # We are going to make a small dataframe and update the configuration to noise 100% of the
    # available rows. We will then check that the correct rows were copied with the correct
    # information.
    dummy_data = pd.DataFrame(
        {
            "simulant_id": [str(i) for i in list(range(10))],
            "household_id": [0, 0, 0, 1, 2, 3, 4, 4, 5, 6],
            "housing_type": [
                "Household",
                "Household",
                "Household",
                "College",
                "Military",
                "Household",
                "Household",
                "Household",
                "Household",
                "Household",
            ],
            "age": [10, 10, 10, 10, 10, 17, 50, 50, 50, 50],
            "guardian_1": ["8", "7", "5", "9", "7", "6", np.nan, np.nan, np.nan, np.nan],
            "guardian_2": [
                "9",
                np.nan,
                "8",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "relationship_to_reference_person": ["Fake value"] * 10,
            "year": [2019] * 10,
            "street_name": [
                "Street 0",
                "Street 0",
                "Street 1",
                "Street 1",
                "Street 1",
                "Street 2",
                "Street 3",
                "Street 3",
                "Street 4",
                "Street 5",
            ],
            "street_number": [0, 0, 1, 1, 1, 2, 3, 3, 4, 5],
            "unit_number": [0, 0, 1, 1, 1, 2, 3, 3, 4, 5],
            "city": [
                "City 0",
                "City 0",
                "City 1",
                "City 1",
                "City 1",
                "City 2",
                "City 3",
                "City 3",
                "City 4",
                "City 5",
            ],
            "state": [
                "State 0",
                "State 0",
                "State 1",
                "State 1",
                "State 1",
                "State 2",
                "State 3",
                "State 3",
                "State 4",
                "State 5",
            ],
            "zipcode": [
                00000,
                00000,
                11111,
                11111,
                11111,
                22222,
                33333,
                33333,
                44444,
                55555,
            ],
        }
    )
    # Noise 100% of rows
    overrides = {
        key: 1.0
        for key in get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
            NOISE_TYPES.duplicate_with_guardian.name
        ]
    }
    dataset_name_1 = DATASETS.census.name
    noised = NOISE_TYPES.duplicate_with_guardian(
        dataset_name_1, dummy_data, overrides, RANDOMNESS
    )

    # We know the following since every dependent is duplicated:
    #  - Simulant ids 0, 1, 2, 3, and-5 will all be duplicated
    #  - Simulant ids 5-9 are guardians. The only overlap is simulant id 5,
    #    who is both a dependent and a guardian
    #  - Simulant id 0 and 3 have two guardians, 8 and 9.

    # Check that the correct rows were duplicated. Duplicated returns all instances of True after
    # the first instance
    duplicated = noised.loc[noised["simulant_id"].duplicated()]
    guardians = dummy_data.loc[dummy_data["simulant_id"].isin(dummy_data["guardian_1"])]
    assert len(noised) == len(dummy_data) + len(duplicated)
    assert set(duplicated["simulant_id"].tolist()) == set(["0", "1", "2", "3", "5"])
    # Only duplicate a depedent one time
    assert noised["simulant_id"].value_counts().max() == 2

    # Check address information is copied in new rows
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
