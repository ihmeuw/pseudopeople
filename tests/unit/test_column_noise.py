import random
from string import ascii_lowercase, ascii_uppercase

import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.utilities import get_configuration

RANDOMNESS0 = RandomnessStream(
    key="test_column_noise", clock=lambda: pd.Timestamp("2020-09-01"), seed=0
)
RANDOMNESS1 = RandomnessStream(
    key="test_column_noise", clock=lambda: pd.Timestamp("2020-09-01"), seed=1
)


@pytest.fixture(scope="module")
def dummy_dataset():
    # Add a column of integer strings
    num_simulants = 100_000
    dummy_idx = pd.Index(range(num_simulants))
    integer_series = pd.Series([str(x) for x in range(num_simulants)])
    # Add missing data from `generate_missing_data` function
    missing_idx = pd.Index([x for x in dummy_idx if x % 3 == 0])
    integer_series.loc[missing_idx] = ""
    # Add a column of character strings
    str_length = 6
    character_series = pd.Series(
        [
            "".join(
                random.choice(ascii_lowercase + ascii_uppercase) for _ in range(str_length)
            )
            for _ in range(num_simulants)
        ]
    )
    # Add missing data from `generate_missing_data` function
    character_series.loc[missing_idx] = ""

    return pd.DataFrame({"numbers": integer_series, "characters": character_series})


@pytest.fixture(scope="module")
def categorical_series():
    return pd.Series(
        ["CA", "WA", "FL", "OR", "CO", "TX", "NY", "VA", "AZ", "''"] * 100_000, name="state"
    )


@pytest.fixture(scope="module")
def string_series():
    return pd.Series(
        ["Unit 1A", "1234", "12/31/2020", "a1b2c3", "100000.00", "123-45-6789", ""] * 100_000,
        name="random_strings",
    )


@pytest.fixture(scope="module")
def default_configuration():
    return get_configuration()


def test_generate_missing_data(dummy_dataset):

    config = get_configuration()
    config.update(
        {
            "decennial_census": {
                "zipcode": {
                    "missing_data": {
                        "row_noise_level": 0.25,
                    },
                },
            },
        }
    )
    config = config["decennial_census"]["zipcode"]["missing_data"]
    data = dummy_dataset["numbers"]
    noised_data = _validate_seed_and_noise_data(
        noise_type=NOISE_TYPES.MISSING_DATA, column=data, config=config
    )

    # Calculate newly missing data, ie data that didn't come in as already missing
    orig_non_missing_idx = data.index[(data.notna()) & (data != "")]
    newly_missing_idx = noised_data.index[
        (noised_data.index.isin(orig_non_missing_idx)) & (noised_data == "")
    ]

    # Check for expected noise level
    expected_noise = config["row_noise_level"]
    actual_noise = len(newly_missing_idx) / len(orig_non_missing_idx)
    assert np.isclose(expected_noise, actual_noise, rtol=0.02)

    # Check that un-noised values are unchanged
    not_noised_idx = noised_data.index[noised_data != ""]
    assert "" not in noised_data[not_noised_idx].values
    assert (data[not_noised_idx] == noised_data[not_noised_idx]).all()


def test_incorrect_selection(categorical_series, default_configuration):
    config = default_configuration["decennial_census"]["state"]["incorrect_selection"]
    noised_data = _validate_seed_and_noise_data(
        noise_type=NOISE_TYPES.INCORRECT_SELECTION, column=categorical_series, config=config
    )

    # Check for expected noise level
    expected_noise = config["row_noise_level"]
    # todo: Update when generate_incorrect_selection uses exclusive resampling
    # Get real expected noise to account for possibility of noising with original value
    # Here we have a a possibility of choosing any of the 50 states for our categorical series fixture
    expected_noise = expected_noise * (1 - 1 / 50)
    actual_noise = (noised_data != categorical_series).mean()
    assert np.isclose(expected_noise, actual_noise, rtol=0.02)

    original_empty_idx = categorical_series.index[categorical_series == ""]
    noised_empty_idx = noised_data.index[noised_data == ""]
    pd.testing.assert_index_equal(original_empty_idx, noised_empty_idx)


@pytest.mark.skip(reason="TODO")
def test_generate_within_household_copies():
    pass


@pytest.mark.skip(reason="TODO")
def test_swap_months_and_days():
    pass


@pytest.mark.skip(reason="TODO")
def test_miswrite_zipcodes():
    pass


@pytest.mark.skip(reason="TODO")
def test_miswrite_ages():
    pass


def test_miswrite_numerics(string_series):
    """
    Validates that only numeric characters are noised in a series at a provided noise level.
    """
    config = get_configuration()
    config.update(
        {
            "decennial_census": {
                "street_number": {
                    "numeric_miswriting": {
                        "row_noise_level": 0.4,
                        "token_noise_level": 0.5,
                    },
                },
            },
        }
    )
    config = config["decennial_census"]["street_number"]["numeric_miswriting"]
    p_row_noise = config.row_noise_level
    p_token_noise = config.token_noise_level
    data = string_series
    noised_data = _validate_seed_and_noise_data(
        noise_type=NOISE_TYPES.NUMERIC_MISWRITING, column=data, config=config
    )

    # Get masks for helper groups, each string in categorical string purpose is to mimic possible string types
    empty_str = data == ""
    unit_number = data == "Unit 1A"
    id_number = data == "1234"
    alt_str = data == "a1b2c3"
    income = data == "100000.00"
    date_of_birth = data == "12/31/2020"
    ssn = data == "123-45-6789"
    expected_noise = p_row_noise * p_token_noise

    # Check empty strings havent changed
    assert (noised_data[empty_str] == "").all()

    for i in range(4):  # "1234"
        assert np.isclose(
            expected_noise,
            (data[id_number].str[i] != noised_data[id_number].str[i]).mean(),
            rtol=0.02,
        )
        assert (noised_data[id_number].str[i].str.isdigit()).all()

    for i in range(6):  # "a1b2c3"
        if i % 2 == 0:
            assert (data[alt_str].str[i] == noised_data[alt_str].str[i]).all()
        else:
            assert np.isclose(
                expected_noise,
                (data[alt_str].str[i] != noised_data[alt_str].str[i]).mean(),
                rtol=0.02,
            )
            assert (noised_data[alt_str].str[i].str.isdigit()).all()

    for i in range(7):  # "Unit 1A"
        if i == 5:
            assert np.isclose(
                expected_noise,
                (data[unit_number].str[i] != noised_data[unit_number].str[i]).mean(),
                rtol=0.02,
            )
            assert (noised_data[unit_number].str[i].str.isdigit()).all()
        else:
            assert (data[unit_number].str[i] == noised_data[unit_number].str[i]).all()

    for i in range(9):  # "100000.00"
        if i == 6:
            assert (data[income].str[i] == noised_data[income].str[i]).all()
        else:
            assert np.isclose(
                expected_noise,
                (data[income].str[i] != noised_data[income].str[i]).mean(),
                rtol=0.02,
            )
            assert (noised_data[income].str[i].str.isdigit()).all()

    for i in range(10):  # "12/31/2020"
        if i in [2, 5]:
            assert (data[date_of_birth].str[i] == noised_data[date_of_birth].str[i]).all()
        else:
            assert np.isclose(
                expected_noise,
                (data[date_of_birth].str[i] != noised_data[date_of_birth].str[i]).mean(),
                rtol=0.02,
            )
            assert (noised_data[date_of_birth].str[i].str.isdigit()).all()

    for i in range(11):  # "123-45-6789"
        if i in [3, 6]:
            assert (data[ssn].str[i] == noised_data[ssn].str[i]).all()
        else:
            assert np.isclose(
                expected_noise,
                (data[ssn].str[i] != noised_data[ssn].str[i]).mean(),
                rtol=0.02,
            )
            assert (noised_data[ssn].str[i].str.isdigit()).all()


@pytest.mark.skip(reason="TODO")
def test_generate_nicknames():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_fake_names():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_phonetic_errors():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_ocr_errors():
    pass


@pytest.mark.parametrize(
    "column",
    [
        "numbers",
        "characters",
    ],
)
def test_generate_typographical_errors(dummy_dataset, column):
    data = dummy_dataset[column]
    config = get_configuration()
    config.update(
        {
            "decennial_census": {
                column: {
                    "typographic": {
                        "row_noise_level": 0.1,
                        "token_noise_level": 0.1,
                        "include_original_token_level": 0.1,
                    },
                },
            },
        }
    )
    config = config["decennial_census"][column]["typographic"]
    noised_data = _validate_seed_and_noise_data(
        noise_type=NOISE_TYPES.TYPOGRAPHIC, column=data, config=config
    )

    not_missing_idx = data.index[(data.notna()) & (data != "")]
    check_original = data.loc[not_missing_idx]
    check_noised = noised_data.loc[not_missing_idx]

    # Check for expected noise level
    p_row_noise = config.row_noise_level
    p_token_noise = config.token_noise_level
    str_lengths = check_original.str.len()  # pd.Series
    p_token_not_noised = 1 - p_token_noise
    p_strings_not_noised = p_token_not_noised**str_lengths  # pd.Series
    p_strings_noised = 1 - p_strings_not_noised  # pd.Series
    expected_noise = p_row_noise * p_strings_noised.mean()
    actual_noise = (check_noised != check_original).mean()
    assert np.isclose(expected_noise, actual_noise, rtol=0.06)

    # Check for expected string growth due to keeping original noised token
    assert (check_noised.str.len() >= check_original.str.len()).all()
    p_include_original_token = config.include_original_token_level
    p_token_does_not_increase_string_length = 1 - p_token_noise * p_include_original_token
    p_strings_do_not_increase_length = (
        p_token_does_not_increase_string_length**str_lengths
    )  # pd.Series
    p_strings_increase_length = 1 - p_strings_do_not_increase_length  # pd.Series
    expected_changed_length = p_row_noise * p_strings_increase_length.mean()
    actual_changed_length = (check_noised.str.len() != check_original.str.len()).mean()
    assert np.isclose(expected_changed_length, actual_changed_length, rtol=0.06)

    # Check that we did not touch the missing data
    assert (
        data.loc[~data.index.isin(not_missing_idx)]
        == noised_data.loc[~noised_data.index.isin(not_missing_idx)]
    ).all()


####################
# HELPER FUNCTIONS #
####################


# TODO: refactor this into its own test parameterized by noise functions
def _validate_seed_and_noise_data(noise_type, column, config):
    """Confirms randomness stream behavior and returns the noised data"""
    noised_data = noise_type(column, config, RANDOMNESS0, f"test_{noise_type.name}")
    noised_data_same_seed = noise_type(column, config, RANDOMNESS0, f"test_{noise_type.name}")
    noised_data_different_seed = noise_type(
        column, config, RANDOMNESS1, f"test_{noise_type.name}"
    )

    assert (noised_data != column).any()
    assert (noised_data == noised_data_same_seed).all()
    assert (noised_data != noised_data_different_seed).any()

    return noised_data
