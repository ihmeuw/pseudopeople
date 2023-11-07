import random
from string import ascii_lowercase, ascii_uppercase

import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants.metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASETS
from pseudopeople.utilities import load_ocr_errors_dict, load_phonetic_errors_dict

RANDOMNESS0 = RandomnessStream(
    key="test_column_noise",
    clock=lambda: pd.Timestamp("2020-09-01"),
    seed=0,
    index_map=IndexMap(),
)
RANDOMNESS1 = RandomnessStream(
    key="test_column_noise",
    clock=lambda: pd.Timestamp("2020-09-01"),
    seed=1,
    index_map=IndexMap(),
)


@pytest.fixture(scope="module")
def dummy_dataset():
    num_simulants = 1_000_000
    dummy_idx = pd.Index(range(num_simulants))

    # Add a column of integer strings
    integer_series = pd.Series([str(x) for x in range(num_simulants)])
    # integer_series = pd.Series(["Jenny 867-5309", "foo"]*int(num_simulants/2))
    # Add missing data from `leave_blanks` function
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
    # Add missing data from `leave_blanks` function
    character_series.loc[missing_idx] = ""

    # Add a categorical series state column
    states_list = ["CA", "WA", "FL", "OR", "CO", "TX", "NY", "VA", "AZ", "''"]
    states = pd.Series(states_list * int(num_simulants / len(states_list)))

    # Add age col by converting integer_series
    maximum_age = 120
    ages = integer_series.apply(pd.to_numeric, args=("coerce",))
    ages = ages / ages.max() * (maximum_age + 1)
    ages[ages.isna()] = -1  # temp nan
    ages = ages.astype(int).astype(str)
    ages[ages == "-1"] = ""

    # Add a string_series column of mixed letters and numbers
    string_list = [
        "fo1",
        "fo2",
        "fo3",
        "Unit 1A",
        "1234",
        "12/31/2020",
        "a1b2c3",
        "100000.00",
        "123-45-6789",
        "",
    ]
    string_series = pd.Series(string_list * int(num_simulants / len(string_list)))
    zipcodes = ["12345", "98765", "02468", "13579", ""]
    zipcode_series = pd.Series(zipcodes * int(num_simulants / len(zipcodes)))
    first_names = [
        "Abigail",
        "Catherine",
        "Bill",
        "Fake name",
        "",
    ]
    first_name_series = pd.Series(first_names * int(num_simulants / len(first_names)))
    last_names = ["A last name", "another last name", "other last name", "last name", ""]
    last_name_series = pd.Series(last_names * int(num_simulants / len(last_names)))
    event_date_list = ["01/25/1990", "05/30/1995", "10/01/2000", "12/31/2010", np.nan]
    event_date_series = pd.Series(event_date_list * int(num_simulants / len(event_date_list)))
    date_of_birth_list = ["01/31/1950", "05/01/1990", "10/01/2000", "12/31/2010", np.nan]
    date_of_birth_series = pd.Series(
        date_of_birth_list * int(num_simulants / len(date_of_birth_list))
    )
    copy_age_list = ["10", "20", "30", "40", "50", "60", "70", "80", "", "100"]
    copy_age_series = pd.Series(copy_age_list * int(num_simulants / len(copy_age_list)))

    return pd.DataFrame(
        {
            "numbers": integer_series,
            "characters": character_series,
            "state": states,
            "age": ages,
            "street_number": string_series,
            "zipcode": zipcode_series,
            "first_name": first_name_series,
            "last_name": last_name_series,
            "event_date": event_date_series,
            "date_of_birth": date_of_birth_series,
            "copy_age": copy_age_series,
        }
    )


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


def test_leave_blank(dummy_dataset):
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["zipcode"][
        NOISE_TYPES.leave_blank.name
    ]
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "zipcode": {
                        NOISE_TYPES.leave_blank.name: {
                            Keys.CELL_PROBABILITY: 0.25,
                        },
                    },
                },
            },
        }
    )
    data = dummy_dataset[["numbers"]]
    noised_data, _ = NOISE_TYPES.leave_blank(data, config, RANDOMNESS0, "dataset", "numbers")

    # Calculate newly missing data, ie data that didn't come in as already missing
    data = data.squeeze()
    orig_non_missing_idx = data.index[(data.notna()) & (data != "")]
    newly_missing_idx = noised_data.index[
        (noised_data.index.isin(orig_non_missing_idx)) & (noised_data.isna())
    ]

    # Check for expected noise level
    expected_noise = config[Keys.CELL_PROBABILITY]
    actual_noise = len(newly_missing_idx) / len(orig_non_missing_idx)
    is_close_wrapper(actual_noise, expected_noise, 0.02)

    # Check that un-noised values are unchanged
    not_noised_idx = noised_data.index[noised_data.notna()]
    assert (data[not_noised_idx] == noised_data[not_noised_idx]).all()


def test_choose_wrong_option(dummy_dataset):
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["state"][
        NOISE_TYPES.choose_wrong_option.name
    ]
    data = dummy_dataset[["state"]]
    noised_data, _ = NOISE_TYPES.choose_wrong_option(
        data, config, RANDOMNESS0, "dataset", "state"
    )
    data = data.squeeze()
    # Check for expected noise level
    expected_noise = config[Keys.CELL_PROBABILITY]
    # todo: Update when choose_wrong_options uses exclusive resampling
    # Get real expected noise to account for possibility of noising with original value
    # Here we have a a possibility of choosing any of the 50 states for our categorical series fixture
    actual_noise = (noised_data != data).mean()
    is_close_wrapper(actual_noise, expected_noise, 0.03)

    original_empty_idx = data.index[data == ""]
    noised_empty_idx = noised_data.index[noised_data == ""]
    pd.testing.assert_index_equal(original_empty_idx, noised_empty_idx)


def test_generate_copy_from_household_member(dummy_dataset):
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][
        NOISE_TYPES.copy_from_household_member.name
    ]
    data = dummy_dataset[["age", "copy_age"]]
    noised_data, _ = NOISE_TYPES.copy_from_household_member(
        data, config, RANDOMNESS0, "dataset", "age"
    )

    # Check for expected noise level
    expected_noise = config[Keys.CELL_PROBABILITY]
    original_missing_idx = data.index[data["age"] == ""]
    eligible_for_noise_idx = data.index.difference(original_missing_idx)
    data = data["age"]
    actual_noise = (
        noised_data[eligible_for_noise_idx] != data[eligible_for_noise_idx]
    ).mean()
    is_close_wrapper(actual_noise, expected_noise, 0.02)

    # Noised values should be the same as the copy column
    was_noised_series = noised_data[eligible_for_noise_idx] != data[eligible_for_noise_idx]
    noised_idx = was_noised_series[was_noised_series].index
    assert (
        dummy_dataset.loc[noised_idx, COPY_HOUSEHOLD_MEMBER_COLS["age"]]
        == noised_data.loc[noised_idx]
    ).all()
    not_noised_idx = dummy_dataset.index.difference(noised_idx)
    assert (dummy_dataset.loc[not_noised_idx, "age"] == noised_data.loc[not_noised_idx]).all()


def test_swap_months_and_days(dummy_dataset):
    for col in ["event_date", "date_of_birth"]:
        data = dummy_dataset[[col]]
        if col == "event_date":
            config = get_configuration()[DATASETS.ssa.name][Keys.COLUMN_NOISE][col][
                NOISE_TYPES.swap_month_and_day.name
            ]
            config.update(
                {
                    DATASETS.ssa.name: {
                        Keys.COLUMN_NOISE: {
                            col: {
                                NOISE_TYPES.swap_month_and_day.name: {
                                    Keys.CELL_PROBABILITY: 0.25,
                                },
                            },
                        },
                    },
                }
            )
        else:
            config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE][col][
                NOISE_TYPES.swap_month_and_day.name
            ]
        expected_noise = config[Keys.CELL_PROBABILITY]
        noised_data, _ = NOISE_TYPES.swap_month_and_day(
            data, config, RANDOMNESS0, DATASETS.census.name, col
        )

        # Confirm missing data remains missing
        data = data.squeeze()
        orig_missing = data.isna()
        assert (noised_data[orig_missing].isna()).all()

        assert (data[~orig_missing].str[6:] == noised_data[~orig_missing].str[6:]).all()
        actual_noise = (data[~orig_missing] != noised_data[~orig_missing]).mean()
        is_close_wrapper(actual_noise, expected_noise, 0.02)


def test_write_wrong_zipcode_digits(dummy_dataset):
    dummy_digit_probabilities = [0.3, 0.3, 0.4, 0.5, 0.5]
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "zipcode": {
                        NOISE_TYPES.write_wrong_zipcode_digits.name: {
                            Keys.CELL_PROBABILITY: 0.5,
                            Keys.ZIPCODE_DIGIT_PROBABILITIES: dummy_digit_probabilities,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["zipcode"][
        NOISE_TYPES.write_wrong_zipcode_digits.name
    ]

    # Get configuration values for each piece of 5 digit zipcode
    probability = config[Keys.CELL_PROBABILITY]
    data = dummy_dataset[["zipcode"]]
    noised_data, _ = NOISE_TYPES.write_wrong_zipcode_digits(
        data, config, RANDOMNESS0, "dataset", "zipcode"
    )

    # Confirm missing data remains missing
    data = data.squeeze()
    orig_missing = data == ""
    assert (noised_data[orig_missing] == "").all()
    # Check noise for each digits position matches expected noise
    for i in range(5):
        digit_prob = config["digit_probabilities"][i]
        actual_noise = (
            data[~orig_missing].str[i] != noised_data[~orig_missing].str[i]
        ).mean()
        expected_noise = digit_prob * probability
        is_close_wrapper(actual_noise, expected_noise, 0.005)


def test_miswrite_ages_default_config(dummy_dataset):
    """Test that miswritten ages are appropriately handled, including
    no perturbation probabilities defaults to uniform distribution,
    perturbation probabilities"""
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][
        NOISE_TYPES.misreport_age.name
    ]
    data = dummy_dataset[["age"]]
    noised_data, _ = NOISE_TYPES.misreport_age(data, config, RANDOMNESS0, "dataset", "age")
    data = data.squeeze()

    # Check for expected noise level
    not_missing_idx = data.index[data != ""]
    expected_noise = config[Keys.CELL_PROBABILITY]
    actual_noise = (noised_data[not_missing_idx] != data[not_missing_idx]).mean()
    # NOTE: the expected noise calculated above does not account for the fact that
    # if a perturbed age ends up being the same as the original age, then 1 is subtracted.
    is_close_wrapper(actual_noise, expected_noise, 0.001)

    # Check that missing data remains missing
    original_missing_idx = data.index[data == ""]
    noised_missing_idx = noised_data.index[noised_data == ""]
    pd.testing.assert_index_equal(original_missing_idx, noised_missing_idx)

    # Check that there are no negative ages generated
    assert noised_data[not_missing_idx].astype(int).min() >= 0


def test_miswrite_ages_uniform_probabilities():
    """Test that a list of perturbations passed in results in uniform probabilities"""
    num_rows = 100_000
    original_age = 25
    perturbations = [-2, -1, 1]

    config = get_configuration(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 1,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][NOISE_TYPES.misreport_age.name]

    data = pd.Series([str(original_age)] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    noised_data, _ = NOISE_TYPES.misreport_age(df, config, RANDOMNESS0, "dataset", "age")
    expected_noise = 1 / len(perturbations)
    for perturbation in perturbations:
        actual_noise = (noised_data.astype(int) - original_age == perturbation).mean()
        is_close_wrapper(actual_noise, expected_noise, 0.01)


def test_miswrite_ages_provided_probabilities():
    """Test that provided age perturation probabilites are handled"""
    num_rows = 100_000
    original_age = 25
    perturbations = {-1: 0.1, 1: 0.9}

    config = get_configuration(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 1,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][NOISE_TYPES.misreport_age.name]

    data = pd.Series([str(original_age)] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    noised_data, _ = NOISE_TYPES.misreport_age(df, config, RANDOMNESS0, "dataset", "age")
    for perturbation in perturbations:
        expected_noise = perturbations[perturbation]
        actual_noise = (noised_data.astype(int) - original_age == perturbation).mean()
        is_close_wrapper(actual_noise, expected_noise, 0.02)


def test_miswrite_ages_handles_perturbation_to_same_age():
    """Tests an edge case. It's possible that after an age is perturbed it ends
    up being the original age. In that case, subtract 1. eg, an age of 1 that is
    perturbed -2 becomes -1. But we cannot have negative so we flip the sign to +1.
    But that's the same as the original age and so should become 1-1=0.
    """
    num_rows = 100
    age = 1
    perturbations = [-2]  # This will cause -1 which will be flipped to +1

    config = get_configuration(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 1,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][NOISE_TYPES.misreport_age.name]

    data = pd.Series([str(age)] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    noised_data, _ = NOISE_TYPES.misreport_age(df, config, RANDOMNESS0, "dataset", "age")

    assert (noised_data == 0).all()


def test_miswrite_ages_flips_negative_to_positive():
    """Test that any ages perturbed to <0 are reflected to positive values"""
    num_rows = 100
    age = 3
    perturbations = [-7]  # This will cause -4 and should flip to +4

    config = get_configuration(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 1,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][NOISE_TYPES.misreport_age.name]

    data = pd.Series([str(age)] * num_rows, name="age")
    df = pd.DataFrame({"age": data})
    noised_data, _ = NOISE_TYPES.misreport_age(df, config, RANDOMNESS0, "dataset", "age")

    assert (noised_data == 4).all()


@pytest.mark.slow
def test_write_wrong_digits_robust(dummy_dataset):
    """
    Validates that only numeric characters are noised in a series at a provided noise level.
    """
    # This test is pretty slow because of the number of times we have to iterate through the series
    # so I have marked it as slow - albrja
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "street_number": {
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["street_number"][
        NOISE_TYPES.write_wrong_digits.name
    ]
    p_row_noise = config[Keys.CELL_PROBABILITY]
    p_token_noise = config[Keys.TOKEN_PROBABILITY]
    data = dummy_dataset[["street_number"]]
    # Note: I changed this column from string_series to street number. It has several string formats
    # containing both numeric and alphabetically string characters.
    noised_data, _ = NOISE_TYPES.write_wrong_digits(
        data, config, RANDOMNESS0, "dataset", "street_number"
    )

    # Get masks for helper groups, each string in categorical string purpose is to mimic possible string types
    data = data["street_number"]
    empty_str = data == ""
    ambig_str = data.str.len() == 3
    unit_number = data == "Unit 1A"
    id_number = data == "1234"
    alt_str = data == "a1b2c3"
    income = data == "100000.00"
    date_of_birth = data == "12/31/2020"
    ssn = data == "123-45-6789"
    expected_noise = p_row_noise * p_token_noise

    # Check empty strings havent changed
    assert (noised_data[empty_str] == "").all()
    # Assert string length doesn't change after noising
    assert (data.str.len() == noised_data.str.len()).all()

    for i in range(3):  # "fo1", "fo2", "fo3"
        if i == 2:
            actual_noise = (data[ambig_str].str[i] != noised_data[ambig_str].str[i]).mean()
            is_close_wrapper(actual_noise, expected_noise, 0.02)
        else:
            assert (data[ambig_str].str[i] == noised_data[ambig_str].str[i]).all()

    for i in range(4):  # "1234"
        actual_noise = (data[id_number].str[i] != noised_data[id_number].str[i]).mean()
        is_close_wrapper(actual_noise, expected_noise, 0.02)
        assert (noised_data[id_number].str[i].str.isdigit()).all()

    for i in range(6):  # "a1b2c3"
        if i % 2 == 0:
            assert (data[alt_str].str[i] == noised_data[alt_str].str[i]).all()
        else:
            actual_noise = (data[alt_str].str[i] != noised_data[alt_str].str[i]).mean()
            is_close_wrapper(actual_noise, expected_noise, 0.02)
            assert (noised_data[alt_str].str[i].str.isdigit()).all()

    for i in range(7):  # "Unit 1A"
        if i == 5:
            actual_noise = (
                data[unit_number].str[i] != noised_data[unit_number].str[i]
            ).mean()
            is_close_wrapper(actual_noise, expected_noise, 0.02)
            assert (noised_data[unit_number].str[i].str.isdigit()).all()
        else:
            assert (data[unit_number].str[i] == noised_data[unit_number].str[i]).all()

    for i in range(9):  # "100000.00"
        if i == 6:
            assert (data[income].str[i] == noised_data[income].str[i]).all()
        else:
            actual_noise = (data[income].str[i] != noised_data[income].str[i]).mean()
            is_close_wrapper(actual_noise, expected_noise, 0.02)
            assert (noised_data[income].str[i].str.isdigit()).all()

    for i in range(10):  # "12/31/2020"
        if i in [2, 5]:
            assert (data[date_of_birth].str[i] == noised_data[date_of_birth].str[i]).all()
        else:
            actual_noise = (
                data[date_of_birth].str[i] != noised_data[date_of_birth].str[i]
            ).mean()
            is_close_wrapper(actual_noise, expected_noise, 0.02)
            assert (noised_data[date_of_birth].str[i].str.isdigit()).all()

    for i in range(11):  # "123-45-6789"
        if i in [3, 6]:
            assert (data[ssn].str[i] == noised_data[ssn].str[i]).all()
        else:
            actual_noise = (data[ssn].str[i] != noised_data[ssn].str[i]).mean()
            is_close_wrapper(actual_noise, expected_noise, 0.02)
            assert (noised_data[ssn].str[i].str.isdigit()).all()


def test_write_wrong_digits(dummy_dataset):
    # This is a quicker (less robust) version of the test above.
    # It only checks that numeric characters are noised at the correct level as
    # a sanity check our noise is of the right magnitude
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "street_number": {
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["street_number"][
        NOISE_TYPES.write_wrong_digits.name
    ]
    expected_cell_noise = config[Keys.CELL_PROBABILITY]

    data = dummy_dataset[["street_number"]]
    # Note: I changed this column from string_series to street number. It has several string formats
    # containing both numeric and alphabetically string characters.
    noised_data, _ = NOISE_TYPES.write_wrong_digits(
        data, config, RANDOMNESS0, "dataset", "street_number"
    )

    # Validate we do not change any missing data
    data = data.squeeze()
    missing_mask = data == ""
    assert (noised_data[missing_mask] == "").all()

    # Check expected noise level
    check_original = data[~missing_mask]
    check_noised = noised_data[~missing_mask]
    actual_noise = (check_original != check_noised).mean()
    # We are setting lower and upper bounds for testing
    assert actual_noise < expected_cell_noise
    assert actual_noise > expected_cell_noise / 10


def test_use_nickname(dummy_dataset):
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["first_name"][
        NOISE_TYPES.use_nickname.name
    ]
    expected_noise = config[Keys.CELL_PROBABILITY]
    data = dummy_dataset[["first_name"]]
    noised_data, _ = NOISE_TYPES.use_nickname(
        data, config, RANDOMNESS0, "dataset", "first_name"
    )
    data = data.squeeze()

    # Validate missing stays missing
    orig_missing = data.isna()
    assert (noised_data[orig_missing].isna()).all()
    # Validate noise level
    actual_noise = (noised_data[~orig_missing] != data[~orig_missing]).mean()
    is_close_wrapper(actual_noise, expected_noise, 0.02)

    # Validation for nicknames
    from pseudopeople.noise_scaling import load_nicknames_data

    nicknames = load_nicknames_data()
    names_list = pd.Series(
        nicknames.apply(lambda row: row.dropna().tolist(), axis=1), index=nicknames.index
    )
    for real_name in data.dropna().unique():
        # Validates names that are not nickname eligible do not not get noised
        if real_name not in names_list.index:
            assert (data.loc[data == real_name] == noised_data[data == real_name]).all()
        else:
            real_name_idx = data.index[data == real_name]
            # Verify options chosen are valid nicknames for original names that were noised
            assert set(noised_data.loc[real_name_idx].dropna()).issubset(
                set(names_list.loc[real_name] + [real_name])
            )
            # Validate we choose the nicknames for each name randomly (equally)
            chosen_nicknames = noised_data.loc[
                real_name_idx.difference(noised_data.index[noised_data == real_name])
            ]
            chosen_nickname_weights = pd.Series(
                chosen_nicknames.value_counts() / sum(chosen_nicknames.value_counts())
            )
            name_weight = 1 / len(names_list.loc[real_name])
            # We are weighting our rtol to adjust for variance depending on number of nicknames
            rtol = 0.025 * len(chosen_nickname_weights)
            is_close_wrapper(chosen_nickname_weights, name_weight, rtol)


def test_use_fake_name(dummy_dataset):
    """
    Function to test that fake names are noised and replace raw values at a configured percentage
    """
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "first_name": {
                        NOISE_TYPES.use_fake_name.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                        },
                    },
                    "last_name": {
                        NOISE_TYPES.use_fake_name.name: {Keys.CELL_PROBABILITY: 0.5}
                    },
                },
            },
        }
    )
    first_name_config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["first_name"][
        NOISE_TYPES.use_fake_name.name
    ]
    last_name_config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["last_name"][
        NOISE_TYPES.use_fake_name.name
    ]

    # For this test, using the dummy_dataset fixture the "string_series" column will be used as both names columns
    # This will help demonstrate that the additional key is working correctly
    first_name_data = dummy_dataset[["first_name"]]
    last_name_data = dummy_dataset[["last_name"]]
    noised_first_names, _ = NOISE_TYPES.use_fake_name(
        first_name_data, first_name_config, RANDOMNESS0, "dataset", "first_name"
    )
    noised_last_names, _ = NOISE_TYPES.use_fake_name(
        last_name_data, last_name_config, RANDOMNESS0, "dataset", "last_name"
    )
    first_name_data = first_name_data.squeeze()
    last_name_data = last_name_data.squeeze()

    # Check missing are unchanged
    orig_missing = first_name_data == ""
    assert (first_name_data[orig_missing] == noised_first_names[orig_missing]).all()
    assert (last_name_data[orig_missing] == noised_last_names[orig_missing]).all()
    # todo: equal across fake values
    # Check noised values
    actual_first_name_noise = (
        first_name_data[~orig_missing] != noised_first_names[~orig_missing]
    ).mean()
    expected_first_name_noise = first_name_config[Keys.CELL_PROBABILITY]
    is_close_wrapper(actual_first_name_noise, expected_first_name_noise, 0.002)

    actual_last_name_noise = (
        last_name_data[~orig_missing] != noised_last_names[~orig_missing]
    ).mean()
    expected_last_name_noise = last_name_config[Keys.CELL_PROBABILITY]
    is_close_wrapper(actual_last_name_noise, expected_last_name_noise, 0.002)

    # Get raw fake names lists to check noised values
    fake_first = fake_first_names
    fake_last = fake_last_names
    assert (
        noised_first_names.loc[noised_first_names != first_name_data].isin(fake_first).all()
    )
    assert noised_last_names.loc[noised_last_names != last_name_data].isin(fake_last).all()


@pytest.mark.parametrize(
    "column",
    [
        "first_name",
        "last_name",
    ],
)
def test_generate_phonetic_errors(dummy_dataset, column):
    data = dummy_dataset[[column]]

    config = get_configuration(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        NOISE_TYPES.make_phonetic_errors.name: {
                            Keys.CELL_PROBABILITY: 0.1,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    }
                },
            },
        }
    )
    # Get node
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE][column][
        NOISE_TYPES.make_phonetic_errors.name
    ]
    noised_data, _ = NOISE_TYPES.make_phonetic_errors(
        dummy_dataset, config, RANDOMNESS0, "dataset", column
    )
    data = data.squeeze()

    # Validate we do not change any missing data
    missing_mask = data.isna()
    assert noised_data[missing_mask].isna().all()

    # Check expected noise level
    cell_prob = config[Keys.CELL_PROBABILITY]
    check_original = data[~missing_mask]
    check_noised = noised_data[~missing_mask]
    actual_noise = (check_original != check_noised).mean()
    # We are setting lower and upper bounds for testing
    assert actual_noise < cell_prob
    assert actual_noise > cell_prob / 10


def test_phonetic_error_values():
    phonetic_errors_dict = load_phonetic_errors_dict()
    data = pd.Series(list(phonetic_errors_dict.keys()) * 100, name="street_name")
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "street_name": {
                        NOISE_TYPES.make_phonetic_errors.name: {
                            Keys.CELL_PROBABILITY: 1.0,
                            Keys.TOKEN_PROBABILITY: 1.0,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["street_name"][
        NOISE_TYPES.make_phonetic_errors.name
    ]
    df = pd.DataFrame({"street_name": data})
    noised_data, _ = NOISE_TYPES.make_phonetic_errors(
        df, config, RANDOMNESS0, "dataset", "street_name"
    )

    for key in phonetic_errors_dict.keys():
        key_idx = data.index[data == key]
        noised_values = set(noised_data.loc[key_idx])
        pho_error_values = set(phonetic_errors_dict[key])
        assert noised_values == pho_error_values

    assert (data != noised_data).all()


@pytest.mark.parametrize(
    "column",
    [
        "numbers",
        "characters",
    ],
)
def test_generate_ocr_errors(dummy_dataset, column):
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        NOISE_TYPES.make_ocr_errors.name: {
                            Keys.CELL_PROBABILITY: 0.1,
                            Keys.TOKEN_PROBABILITY: 1.0,
                        },
                    }
                },
            },
        }
    )
    # Get node
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE][column][
        NOISE_TYPES.make_ocr_errors.name
    ]
    data = dummy_dataset[[column]]
    noised_data, _ = NOISE_TYPES.make_ocr_errors(data, config, RANDOMNESS0, "dataset", column)
    data = data.squeeze()

    # Validate we do not change any missing data
    missing_mask = data == ""
    assert (data[missing_mask] == noised_data[missing_mask]).all()

    # Check expected noise level
    token_prob = config[Keys.TOKEN_PROBABILITY]
    cell_prob = config[Keys.CELL_PROBABILITY]

    check_original = data[~missing_mask]
    check_noised = noised_data[~missing_mask]
    str_lengths = check_original.str.len() * 2 - 1
    p_token_not_noised = 1 - token_prob  # pd.Series
    # Get probability no tokens are noised in a string
    p_strings_not_noised = p_token_not_noised**str_lengths  # pd.Series
    p_strings_noised = 1 - p_strings_not_noised  # pd.Series
    upper_bound = cell_prob * p_strings_noised.mean()
    actual_noise = (check_original != check_noised).mean()
    # We have simplified the expected noise calculation. Note that not all tokens are eligible for OCR noising so we
    # should never meet our upper bound of expected noise. Alternatively, we want to make sure the noise level is not
    # unexpectedly small.
    assert actual_noise < upper_bound
    assert actual_noise > upper_bound / 10


def test_ocr_replacement_values():
    # Test that OCR noising replaces truth value with correct error values
    # Load OCR errors dict
    ocr_errors_dict = load_ocr_errors_dict()
    # Make series of OCR error dict keys - is there an intelligent numberto pick besides 10?
    data = pd.Series(list(ocr_errors_dict.keys()) * 10, name="employer_name")
    df = pd.DataFrame({"employer_name": data})
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "employer_name": {
                        NOISE_TYPES.make_ocr_errors.name: {
                            Keys.CELL_PROBABILITY: 1.0,
                            Keys.TOKEN_PROBABILITY: 1.0,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["employer_name"][
        NOISE_TYPES.make_ocr_errors.name
    ]
    noised_data, _ = NOISE_TYPES.make_ocr_errors(
        df, config, RANDOMNESS0, "dataset", "employer_name"
    )

    for key in ocr_errors_dict.keys():
        key_idx = data.index[data == key]
        noised_values = set(noised_data.loc[key_idx])
        ocr_error_values = set(ocr_errors_dict[key])
        assert noised_values == ocr_error_values

    assert (data != noised_data).all()


@pytest.mark.parametrize(
    "column",
    [
        "numbers",
        "characters",
    ],
)
def test_make_typos(dummy_dataset, column):
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        NOISE_TYPES.make_typos.name: {
                            Keys.CELL_PROBABILITY: 0.1,
                            Keys.TOKEN_PROBABILITY: 0.1,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE][column][
        NOISE_TYPES.make_typos.name
    ]
    data = dummy_dataset[[column]]
    noised_data, _ = NOISE_TYPES.make_typos(data, config, RANDOMNESS0, "dataset", column)
    data = data.squeeze()

    not_missing_idx = data.index[(data.notna()) & (data != "")]
    check_original = data.loc[not_missing_idx]
    check_noised = noised_data.loc[not_missing_idx]

    # Check for expected noise level
    p_row_noise = config[Keys.CELL_PROBABILITY]
    p_token_noise = config[Keys.TOKEN_PROBABILITY]
    str_lengths = check_original.str.len()  # pd.Series
    p_token_not_noised = 1 - p_token_noise
    p_strings_not_noised = p_token_not_noised**str_lengths  # pd.Series
    p_strings_noised = 1 - p_strings_not_noised  # pd.Series
    expected_noise = p_row_noise * p_strings_noised.mean()
    actual_noise = (check_noised != check_original).mean()
    is_close_wrapper(actual_noise, expected_noise, 0.011)

    # Check for expected string growth due to keeping original noised token
    assert (check_noised.str.len() >= check_original.str.len()).all()
    # TODO: remove this hard-coding
    p_include_original_token = 0.1
    p_token_does_not_increase_string_length = 1 - p_token_noise * p_include_original_token
    p_strings_do_not_increase_length = (
        p_token_does_not_increase_string_length**str_lengths
    )  # pd.Series
    p_strings_increase_length = 1 - p_strings_do_not_increase_length  # pd.Series
    expected_changed_length = p_row_noise * p_strings_increase_length.mean()
    actual_changed_length = (check_noised.str.len() != check_original.str.len()).mean()
    is_close_wrapper(actual_changed_length, expected_changed_length, 0.05)

    # Check that we did not touch the missing data
    assert (
        data.loc[~data.index.isin(not_missing_idx)]
        == noised_data.loc[~noised_data.index.isin(not_missing_idx)]
    ).all()


@pytest.mark.parametrize(
    "noise_type, data_col, dataset, dataset_col",
    [
        (NOISE_TYPES.leave_blank, "numbers", "decennial_census", "zipcode"),
        (NOISE_TYPES.choose_wrong_option, "state", "decennial_census", "state"),
        (NOISE_TYPES.copy_from_household_member, "age", "decennial_census", "age"),
        (NOISE_TYPES.swap_month_and_day, "event_date", "social_security", "event_date"),
        (NOISE_TYPES.write_wrong_zipcode_digits, "zipcode", "decennial_census", "zipcode"),
        (NOISE_TYPES.misreport_age, "age", "decennial_census", "age"),
        (
            NOISE_TYPES.write_wrong_digits,
            "street_number",
            "decennial_census",
            "street_number",
        ),
        (NOISE_TYPES.use_nickname, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.use_fake_name, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.use_fake_name, "last_name", "decennial_census", "last_name"),
        (NOISE_TYPES.make_phonetic_errors, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.make_ocr_errors, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.make_typos, "numbers", "decennial_census", "zipcode"),
        (NOISE_TYPES.make_typos, "characters", "decennial_census", "street_name"),
    ],
)
def test_seeds_behave_as_expected(noise_type, data_col, dataset, dataset_col, dummy_dataset):
    """Tests that different seeds produce different results and the same seed
    produces the same results
    """
    if data_col == "todo":
        pytest.skip(reason=f"TODO: implement for {noise_type}")
    noise = noise_type.name
    config = get_configuration()[dataset][Keys.COLUMN_NOISE][dataset_col][noise]
    if noise == NOISE_TYPES.copy_from_household_member.name:
        data = dummy_dataset[[data_col, COPY_HOUSEHOLD_MEMBER_COLS[data_col]]]
    else:
        data = dummy_dataset[[data_col]]

    noised_data, _ = noise_type(data, config, RANDOMNESS0, dataset, data_col)
    noised_data_same_seed, _ = noise_type(data, config, RANDOMNESS0, dataset, data_col)
    noised_data_different_seed, _ = noise_type(data, config, RANDOMNESS1, dataset, data_col)
    data = data[data_col]

    assert (noised_data != data).any()
    assert (noised_data.isna() == noised_data_same_seed.isna()).all()
    assert (
        noised_data[noised_data.notna()]
        == noised_data_same_seed[noised_data_same_seed.notna()]
    ).all()
    assert (noised_data != noised_data_different_seed).any()

    # Check that we are in fact getting different noised values
    noised = noised_data.loc[noised_data != data].reset_index(drop=True)
    noised_different_seed = noised_data_different_seed.loc[
        noised_data_different_seed != data
    ].reset_index(drop=True)
    shortest = min(len(noised), len(noised_different_seed))
    assert (noised.iloc[:shortest] != noised_different_seed.iloc[:shortest]).any()


def np_isclose_wrapper(actual_noise, expected_noise, rtol):
    return np.isclose(actual_noise, expected_noise, rtol).all()


def is_close_wrapper(actual_noise, expected_noise, rtol):
    assert np_isclose_wrapper(
        actual_noise,
        expected_noise,
        rtol,
    ), f"Actual noise is {actual_noise} while expected noise was {expected_noise} with a rtol of {rtol}"
