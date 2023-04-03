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
    num_simulants = 1_000_000
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


@pytest.fixture()
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


def test_miswrite_ages(dummy_dataset, default_configuration):
    """Test that miswritten ages are appropriately handled, including
    no perturbation probabilities defaults to uniform distribution,
    perturbation probabilities"""
    config = default_configuration["decennial_census"]["age"]["age_miswriting"]
    data = dummy_dataset.rename(columns={"numbers": "age"})["age"]

    # Convert to realistic age
    maximum_age = 120
    data = data.apply(pd.to_numeric, args=("coerce",))
    data = data / data.max() * (maximum_age + 1)
    data[data.isna()] = -1  # temp nan
    data = data.astype(int).astype(str)
    data[data == "-1"] = ""
    noised_data = _validate_seed_and_noise_data(
        noise_type=NOISE_TYPES.AGE_MISWRITING, column=data, config=config
    )

    # Check for expected noise level
    not_missing_idx = data.index[data != ""]
    expected_noise = config["row_noise_level"]
    # todo: Update when generate_incorrect_selection uses exclusive resampling
    actual_noise = (noised_data[not_missing_idx] != data[not_missing_idx]).mean()
    # NOTE: we increase the relative tolerance a bit here because the expected
    # noise calculated above does not account for the fact that if a perturbed
    # age ends up being the same as the original age, then 1 is subtracted.
    assert np.isclose(expected_noise, actual_noise, rtol=0.03)

    # Check that missing data remains missing
    original_missing_idx = data.index[data == ""]
    noised_missing_idx = noised_data.index[noised_data == ""]
    pd.testing.assert_index_equal(original_missing_idx, noised_missing_idx)

    # Check that there are no negative ages generated
    assert noised_data[not_missing_idx].astype(int).min() >= 0


def test_miswrite_ages_uniform_probabilities(default_configuration):
    """Test that an empty list passed in for probabilities results in uniform"""
    num_rows = 1_000_000
    age = 25
    perturbations = [-2, -1, 1]

    config = default_configuration["decennial_census"]["age"]["age_miswriting"]
    config.update(
        {
            "row_noise_level": 1,
            "possible_perturbations": perturbations,
        },
    )

    data = pd.Series([str(age)] * num_rows)
    noised_data = NOISE_TYPES.AGE_MISWRITING(data, config, RANDOMNESS0, "test")
    expected_noise = 1 / len(perturbations)
    for perturbed_age in [str(x + 25) for x in perturbations]:
        actual_noise = (noised_data == perturbed_age).mean()
        assert np.isclose(actual_noise, expected_noise, rtol=0.01)


def test_miswrite_ages_provided_probabilities(default_configuration):
    """Test that provided age perturation probabilites are handled"""
    num_rows = 1_000_000
    age = 25
    perturbations = [-1, 1]
    probabilities = [0.1, 0.9]

    config = default_configuration["decennial_census"]["age"]["age_miswriting"]
    config.update(
        {
            "row_noise_level": 1,
            "possible_perturbations": perturbations,
            "possible_perturbation_levels": probabilities,
        },
    )

    data = pd.Series([str(age)] * num_rows)
    noised_data = NOISE_TYPES.AGE_MISWRITING(data, config, RANDOMNESS0, "test")
    for i in range(len(perturbations)):
        expected_noise = probabilities[i]
        perturbed_age = str(age + perturbations[i])
        actual_noise = (noised_data == perturbed_age).mean()
        assert np.isclose(actual_noise, expected_noise, rtol=0.01)


def test_miswrite_ages_handles_perturbation_to_same_age(default_configuration):
    """Tests an edge case. It's possible that after an age is perturbed it ends
    up being the original age. In that case, subtract 1. eg, an age of 1 that is
    perturbed becomes -1. But we cannot have negative so we flip the sign to +1.
    But that's the same as the original age and so should become 1-1=0.
    """
    num_rows = 100
    age = 1
    perturbations = [-1]

    config = default_configuration["decennial_census"]["age"]["age_miswriting"]
    config.update(
        {
            "row_noise_level": 1,
            "possible_perturbations": perturbations,
        },
    )

    data = pd.Series([str(age)] * num_rows)
    noised_data = NOISE_TYPES.AGE_MISWRITING(data, config, RANDOMNESS0, "test")

    assert (noised_data == "0").all()


def test_miswrite_ages_fails_if_bad_probs(default_configuration):
    """Test that runtimerrors if probs do not add up to 1"""
    probabilities = [0.1, 0.8]  # does not sum to 1

    config = default_configuration["decennial_census"]["age"]["age_miswriting"]
    config.update(
        {
            "possible_perturbation_levels": probabilities,
        },
    )

    data = pd.Series(["dummy"])
    with pytest.raises(Exception):
        NOISE_TYPES.AGE_MISWRITING(data, config, RANDOMNESS0, "test")


def test_miswrite_ages_fails_if_incorrect_number_of_probs(default_configuration):
    """Test that runtimerrors if the length of the provided probabilies does
    not match the length of the possible perturbations
    """
    probabilities = [0.1, 0.2, 0.7]  # default perturbations are [-1, 1]

    config = default_configuration["decennial_census"]["age"]["age_miswriting"]
    config.update(
        {
            "possible_perturbation_levels": probabilities,
        },
    )

    data = pd.Series(["dummy"])
    with pytest.raises(Exception):
        NOISE_TYPES.AGE_MISWRITING(data, config, RANDOMNESS0, "test")


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
def test_generate_typographical_errors(dummy_dataset, column, default_configuration):
    data = dummy_dataset[column]
    config = default_configuration
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
