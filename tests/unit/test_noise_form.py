import random
from string import ascii_lowercase
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType
from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise import noise_dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASETS


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


@pytest.fixture(scope="module")
def dummy_config_noise_numbers():
    """Create a dummy configuration that applies all noise functions to a single
    column in the dummy_data fixture. All noise function specs are defined in
    reverse order here compared to how they are to be applied.

    NOTE: this is not a realistic scenario but allows for certain
    types of stress testing.
    """
    return ConfigTree(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "event_type": {
                        NOISE_TYPES.missing_data.name: {Keys.PROBABILITY: 0.01},
                        NOISE_TYPES.incorrect_selection.name: {Keys.PROBABILITY: 0.01},
                        "copy_from_within_household": {Keys.PROBABILITY: 0.01},
                        "month_day_swap": {Keys.PROBABILITY: 0.01},
                        NOISE_TYPES.zipcode_miswriting.name: {
                            Keys.PROBABILITY: 0.01,
                            Keys.ZIPCODE_DIGIT_PROBABILITIES: [0.04, 0.04, 0.2, 0.36, 0.36],
                        },
                        NOISE_TYPES.age_miswriting.name: {
                            Keys.PROBABILITY: 0.01,
                            Keys.POSSIBLE_AGE_DIFFERENCES: [1, -1],
                        },
                        NOISE_TYPES.numeric_miswriting.name: {
                            Keys.PROBABILITY: 0.01,
                            "numeric_miswriting": [0.1],
                        },
                        "nickname": {Keys.PROBABILITY: 0.01},
                        NOISE_TYPES.fake_name.name: {Keys.PROBABILITY: 0.01},
                        "phonetic": {
                            Keys.PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        "ocr": {
                            Keys.PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                        NOISE_TYPES.typographic.name: {
                            Keys.PROBABILITY: 0.01,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                    },
                },
                Keys.ROW_NOISE: {
                    "duplication": {
                        Keys.PROBABILITY: 0.01,
                    },
                    NOISE_TYPES.omission.name: {
                        Keys.PROBABILITY: 0.01,
                    },
                },
            },
        }
    )


def test_noise_order(mocker, dummy_data, dummy_config_noise_numbers):
    """From docs: "Noising should be applied in the following order: omissions, duplications,
    missing data, incorrect selection, copy from w/in household, month and day
    swaps, zip code miswriting, age miswriting, numeric miswriting, nicknames,
    fake names, phonetic, OCR, typographic"
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
            if field in ["omission", "duplication"]
            else dummy_data["event_type"]
        )
        mock.attach_mock(
            mocker.patch(
                f"pseudopeople.noise.NOISE_TYPES.{field}.noise_function",
                return_value=mock_return,
            ),
            field,
        )

    # FIXME: would be better to mock the dataset instead of using census
    noise_dataset(DATASETS.census, dummy_data, dummy_config_noise_numbers, 0)

    call_order = [x[0] for x in mock.mock_calls if not x[0].startswith("__")]
    expected_call_order = [
        "omission",
        # "duplication",
        "missing_data",
        "incorrect_selection",
        # "copy_from_within_household",
        # "month_day_swap",
        "zipcode_miswriting",
        "age_miswriting",
        "numeric_miswriting",
        # "nickname",
        "fake_name",
        # "phonetic",
        # "ocr",
        "typographic",
    ]

    assert expected_call_order == call_order


# TODO: beef this function up
def test_columns_noised(dummy_data):
    """Test that the noise functions are only applied to the numbers column
    (as specified in the dummy config)
    """
    config = ConfigTree(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "event_type": {
                        NOISE_TYPES.missing_data.name: {Keys.PROBABILITY: 0.1},
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
        ("todo", "DATASETS.tax_1040"),
    ],
)
def test_correct_datasets_are_used(func, dataset, mocker):
    """Test that each interface noise function uses the correct dataset"""
    if func == "todo":
        pytest.skip(reason=f"TODO: implement function for dataset {dataset}")
    mock = mocker.patch("pseudopeople.interface._generate_dataset")
    _ = func()

    assert mock.call_args[0][0] == dataset


def test_two_noise_functions_are_independent(mocker):
    # Make simple config tree to test 2 noise functions work together
    config_tree = ConfigTree(
        {
            DATASETS.census.name: {
                "column_noise": {
                    "fake_column_one": {
                        "alpha": {Keys.PROBABILITY: 0.20},
                        "beta": {Keys.PROBABILITY: 0.30},
                    },
                    "fake_column_two": {
                        "alpha": {Keys.PROBABILITY: 0.40},
                        "beta": {Keys.PROBABILITY: 0.50},
                    },
                },
            }
        }
    )

    # Mock objects for testing

    class MockNoiseTypes(NamedTuple):
        ALPHA: ColumnNoiseType = ColumnNoiseType(
            "alpha", lambda column, *_: column.str.cat(pd.Series("abc", index=column.index))
        )
        BETA: ColumnNoiseType = ColumnNoiseType(
            "beta", lambda column, *_: column.str.cat(pd.Series("123", index=column.index))
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
        config_tree.decennial_census.column_noise.fake_column_one.alpha[Keys.PROBABILITY]
    )
    col2_expected_abc_proportion = (
        config_tree.decennial_census.column_noise.fake_column_two.alpha[Keys.PROBABILITY]
    )
    col1_expected_123_proportion = (
        config_tree.decennial_census.column_noise.fake_column_one.beta[Keys.PROBABILITY]
    )
    col2_expected_123_proportion = (
        config_tree.decennial_census.column_noise.fake_column_two.beta[Keys.PROBABILITY]
    )

    assert np.isclose(
        noised_data["fake_column_one"].str.contains("abc").mean(),
        col1_expected_abc_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_two"].str.contains("abc").mean(),
        col2_expected_abc_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_one"].str.contains("123").mean(),
        col1_expected_123_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_two"].str.contains("123").mean(),
        col2_expected_123_proportion,
        rtol=0.01,
    )

    # Assert columns experience both noise
    assert np.isclose(
        noised_data["fake_column_one"].str.contains("abc123").mean(),
        col1_expected_abc_proportion * col1_expected_123_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_two"].str.contains("abc123").mean(),
        col2_expected_abc_proportion * col2_expected_123_proportion,
        rtol=0.01,
    )
    assert noised_data["fake_column_one"].str.contains("123abc").sum() == 0
    assert noised_data["fake_column_two"].str.contains("123abc").sum() == 0
