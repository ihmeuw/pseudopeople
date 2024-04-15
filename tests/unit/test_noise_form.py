import random
from string import ascii_lowercase
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType
from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise import noise_dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASETS
from tests.conftest import FuzzyChecker


@pytest.fixture(scope="module")
def dummy_data():
    """Create a two-column dummy dataset"""
    random.seed(0)
    num_rows = 1_000_000
    return pd.DataFrame(
        {
            "event_type": [str(x) for x in range(num_rows)],
            "words": [
                "".join(random.choice(ascii_lowercase) for _ in range(4))
                for _ in range(num_rows)
            ],
        }
    )


def get_dummy_config_noise_numbers(dataset):
    """Create a dummy configuration that applies all noise functions to a single
    column in the dummy_data fixture. All noise function specs are defined in
    reverse order here compared to how they are to be applied.

    NOTE: this is not a realistic scenario but allows for certain
    types of stress testing.
    """
    return LayeredConfigTree(
        {
            dataset.name: {
                Keys.COLUMN_NOISE: {
                    "event_type": {
                        NOISE_TYPES.leave_blank.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.choose_wrong_option.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.copy_from_household_member.name: {
                            Keys.CELL_PROBABILITY: 0.01
                        },
                        NOISE_TYPES.swap_month_and_day.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.write_wrong_zipcode_digits.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.ZIPCODE_DIGIT_PROBABILITIES: [0.04, 0.04, 0.2, 0.36, 0.36],
                        },
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.POSSIBLE_AGE_DIFFERENCES: [1, -1],
                        },
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        NOISE_TYPES.use_nickname.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.use_fake_name.name: {Keys.CELL_PROBABILITY: 0.01},
                        NOISE_TYPES.make_phonetic_errors.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        NOISE_TYPES.make_ocr_errors.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        NOISE_TYPES.make_typos.name: {
                            Keys.CELL_PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                    },
                },
                Keys.ROW_NOISE: {
                    NOISE_TYPES.do_not_respond.name: {
                        Keys.ROW_PROBABILITY: 0.01,
                    },
                    NOISE_TYPES.omit_row.name: {
                        Keys.ROW_PROBABILITY: 0.01,
                    },
                    NOISE_TYPES.duplicate_with_guardian.name: {
                        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0.05,
                        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0.05,
                    },
                },
            },
        }
    )


@pytest.mark.parametrize(
    "dataset",
    list(DATASETS),
)
def test_noise_order(mocker, dummy_data, dataset):
    """From docs: "Noising should be applied in the following order: omit_row,
    do_not_respond, duplicate_row, leave_blank, choose_wrong_option,
    copy_from_household_member, swap_month_and_day, write_wrong_zipcode_digits,
    misreport_age, write_wrong_digits, use_nickname, use_fake_name,
    make_phonetic_errors, make_ocr_errors, make_typos
    """
    mock = mocker.MagicMock()
    # Mock the noise_functions functions so that they are not actually called and
    # return the original one-column dataframe (so that it doesn't become a mock
    # object itself after the first mocked function is applied.)
    mocker.patch(
        "pseudopeople.entity_types.get_index_to_noise", return_value=dummy_data.index
    )
    for field in NOISE_TYPES._fields:
        mock_return = (
            dummy_data[["event_type"]]
            if field
            in [
                NOISE_TYPES.do_not_respond.name,
                NOISE_TYPES.omit_row.name,
                "duplicate_row",
                NOISE_TYPES.duplicate_with_guardian.name,
            ]
            else dummy_data["event_type"]
        )
        mock.attach_mock(
            mocker.patch(
                f"pseudopeople.noise.NOISE_TYPES.{field}.noise_function",
                return_value=mock_return,
            ),
            field,
        )
        if field not in [
            NOISE_TYPES.do_not_respond.name,
            NOISE_TYPES.omit_row.name,
            "duplicate_row",
            NOISE_TYPES.duplicate_with_guardian.name,
        ]:
            mock.attach_mock(
                mocker.patch(
                    f"pseudopeople.noise.NOISE_TYPES.{field}.additional_column_getter",
                    return_value=[],
                ),
                field,
            )
            mock.attach_mock(
                mocker.patch(
                    f"pseudopeople.noise.NOISE_TYPES.{field}.noise_level_scaling_function",
                    return_value=1,
                ),
                field,
            )

    # Get config for dataset
    dummy_config = get_dummy_config_noise_numbers(dataset)
    # FIXME: would be better to mock the dataset instead of using census
    noise_dataset(dataset, dummy_data, dummy_config, 0)

    # This is getting the string of each noise type. There are two mock calls
    # being made to each noise type with how we are mocking noise type attirbutes
    # above causing duplicates in the call list. Call order is each instance a noise
    # function is called. Here we grab the string of the noise type for one mock method
    # call and not the second method.
    call_order = [x[0] for x in mock.mock_calls if type(x[1][0]) == str]
    row_order = [
        noise_type
        for noise_type in NOISE_TYPES._fields
        if noise_type
        in [
            NOISE_TYPES.duplicate_with_guardian.name,
            NOISE_TYPES.do_not_respond.name,
            NOISE_TYPES.omit_row.name,
            "duplicate_row",
        ]
    ]
    column_order = [
        NOISE_TYPES.leave_blank.name,
        NOISE_TYPES.choose_wrong_option.name,
        NOISE_TYPES.copy_from_household_member.name,
        NOISE_TYPES.swap_month_and_day.name,
        NOISE_TYPES.write_wrong_zipcode_digits.name,
        NOISE_TYPES.misreport_age.name,
        NOISE_TYPES.write_wrong_digits.name,
        NOISE_TYPES.use_nickname.name,
        NOISE_TYPES.use_fake_name.name,
        NOISE_TYPES.make_phonetic_errors.name,
        NOISE_TYPES.make_ocr_errors.name,
        NOISE_TYPES.make_typos.name,
    ]
    expected_call_order = row_order + column_order

    assert expected_call_order == call_order


# TODO: beef this function up
def test_columns_noised(dummy_data):
    """Test that the noise functions are only applied to the numbers column
    (as specified in the dummy config)
    """
    config = LayeredConfigTree(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "event_type": {
                        NOISE_TYPES.leave_blank.name: {Keys.CELL_PROBABILITY: 0.1},
                    },
                },
            },
        },
    )
    noised_data = dummy_data.copy()
    noised_data = noise_dataset(DATASETS.census, noised_data, config, 0)

    assert (dummy_data["event_type"] != noised_data["event_type"]).any()
    assert (dummy_data["words"] == noised_data["words"]).all()


@pytest.mark.parametrize(
    "func, dataset",
    [
        (generate_decennial_census, DATASETS.census),
        (generate_american_community_survey, DATASETS.acs),
        (generate_current_population_survey, DATASETS.cps),
        (generate_women_infants_and_children, DATASETS.wic),
        (generate_social_security, DATASETS.ssa),
        (generate_taxes_w2_and_1099, DATASETS.tax_w2_1099),
        (generate_taxes_1040, DATASETS.tax_1040),
    ],
)
def test_correct_datasets_are_used(func, dataset, mocker):
    """Test that each interface noise function uses the correct dataset"""
    if func == "todo":
        pytest.skip(reason=f"TODO: implement function for dataset {dataset}")
    mock = mocker.patch("pseudopeople.interface._generate_dataset")
    _ = func()

    assert mock.call_args[0][0] == dataset


def test_two_noise_functions_are_independent(mocker, fuzzy_checker: FuzzyChecker):
    # Make simple config tree to test 2 noise functions work together
    config_tree = LayeredConfigTree(
        {
            DATASETS.census.name: {
                "column_noise": {
                    "fake_column_one": {
                        "alpha": {Keys.CELL_PROBABILITY: 0.20},
                        "beta": {Keys.CELL_PROBABILITY: 0.30},
                        "leave_blank": {Keys.CELL_PROBABILITY: 0},
                    },
                    "fake_column_two": {
                        "alpha": {Keys.CELL_PROBABILITY: 0.40},
                        "beta": {Keys.CELL_PROBABILITY: 0.50},
                        "leave_blank": {Keys.CELL_PROBABILITY: 0},
                    },
                },
            }
        }
    )

    # Mock objects for testing

    class MockNoiseTypes(NamedTuple):
        ALPHA: ColumnNoiseType = ColumnNoiseType(
            "alpha",
            lambda data, *_: data.squeeze().str.cat(pd.Series("abc", index=data.index)),
        )
        BETA: ColumnNoiseType = ColumnNoiseType(
            "beta",
            lambda data, *_: data.squeeze().str.cat(pd.Series("123", index=data.index)),
        )
        leave_blank = ColumnNoiseType(
            "leave_blank",
            lambda data, *_: pd.Series(np.nan, index=data.index),
        )

    mock_noise_types = MockNoiseTypes()

    mocker.patch("pseudopeople.noise.NOISE_TYPES", mock_noise_types)
    dummy_dataset = pd.DataFrame(
        {
            "fake_column_one": ["cat", "dog", "bird", "bunny", "duck"] * 20_000,
            "fake_column_two": ["shoe", "pants", "shirt", "hat", "sunglasses"] * 20_000,
        }
    )

    noised_data = noise_dataset(
        dataset=DATASETS.census,
        dataset_data=dummy_dataset,
        seed=0,
        configuration=config_tree,
    )

    # Get config values for testing
    col1_expected_abc_proportion = (
        config_tree.decennial_census.column_noise.fake_column_one.alpha[Keys.CELL_PROBABILITY]
    )
    col2_expected_abc_proportion = (
        config_tree.decennial_census.column_noise.fake_column_two.alpha[Keys.CELL_PROBABILITY]
    )
    col1_expected_123_proportion = (
        config_tree.decennial_census.column_noise.fake_column_one.beta[Keys.CELL_PROBABILITY]
    )
    col2_expected_123_proportion = (
        config_tree.decennial_census.column_noise.fake_column_two.beta[Keys.CELL_PROBABILITY]
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_one_abc_proportion",
        observed_numerator=noised_data["fake_column_one"].str.contains("abc").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col1_expected_abc_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_two_abc_proportion",
        observed_numerator=noised_data["fake_column_two"].str.contains("abc").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col2_expected_abc_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_one_123_proportion",
        observed_numerator=noised_data["fake_column_one"].str.contains("123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col1_expected_123_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_two_123_proportion",
        observed_numerator=noised_data["fake_column_two"].str.contains("123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col2_expected_123_proportion,
    )

    # Assert columns experience both noise
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_one_abc123_proportion",
        observed_numerator=noised_data["fake_column_one"].str.contains("abc123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col1_expected_abc_proportion * col1_expected_123_proportion,
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="fake_column_two_abc123_proportion",
        observed_numerator=noised_data["fake_column_two"].str.contains("abc123").sum(),
        observed_denominator=len(noised_data),
        target_proportion=col2_expected_abc_proportion * col2_expected_123_proportion,
    )
    # Assert expected order of noise application
    assert noised_data["fake_column_one"].str.contains("123abc").sum() == 0
    assert noised_data["fake_column_two"].str.contains("123abc").sum() == 0
