import random
from string import ascii_lowercase, ascii_uppercase

import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.noise_functions import (
    generate_fake_names,
    generate_incorrect_selections,
    generate_missing_data,
    generate_nicknames,
    generate_ocr_errors,
    generate_phonetic_errors,
    generate_typographical_errors,
    generate_within_household_copies,
    miswrite_ages,
    miswrite_numerics,
    miswrite_zipcodes,
    swap_months_and_days,
)
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
    integer_series = pd.Series([str(x) for x in range(num_simulants)])
    # Add missing data from `generate_missing_data` function
    missing_idx = pd.Index([x for x in integer_series.index if x % 3 == 0])
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
    missing_idx = pd.Index([x for x in character_series.index if x % 3 == 0])
    character_series.loc[missing_idx] = ""

    return pd.DataFrame({"numbers": integer_series, "characters": character_series})


def test_generate_missing_data(integer_series, user_config_path):
    config = get_configuration(user_config_path)["decennial_census"]["zipcode"][
        "missing_data"
    ]
    noised_data = _validate_seed_and_noise_data(
        func=generate_missing_data, column=integer_series, config=config
    )

    # Check for expected noise level
    expected_noise = config["row_noise_level"]
    actual_noise = (noised_data == "").mean()
    assert np.isclose(expected_noise, actual_noise, rtol=0.02)

    # Check that un-noised values are unchanged
    not_noised_idx = noised_data.index[noised_data != ""]
    assert "" not in noised_data[not_noised_idx].values
    assert (integer_series[not_noised_idx] == noised_data[not_noised_idx]).all()


@pytest.mark.skip(reason="TODO")
def test_generate_incorrect_selections():
    pass


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


@pytest.mark.skip(reason="TODO")
def test_miswrite_numerics():
    pass


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
        func=generate_typographical_errors, column=data, config=config
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
def _validate_seed_and_noise_data(func, column, config):
    """Confirms randomness stream behavior and returns the noised data"""
    noised_data = func(column, config, RANDOMNESS0, f"test_{func.__name__}")
    noised_data_same_seed = func(column, config, RANDOMNESS0, f"test_{func.__name__}")
    noised_data_different_seed = func(column, config, RANDOMNESS1, f"test_{func.__name__}")

    assert (noised_data != column).any()
    assert (noised_data == noised_data_same_seed).all()
    assert (noised_data != noised_data_different_seed).any()

    return noised_data
