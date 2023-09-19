from typing import Any

import numpy as np
import pandas as pd
import yaml
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream, get_hash

from pseudopeople.configuration import Keys
from pseudopeople.constants import data_values, paths
from pseudopeople.constants.metadata import COPY_HOUSEHOLD_MEMBER_COLS, DatasetNames
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.noise_scaling import load_nicknames_data
from pseudopeople.utilities import (
    get_index_to_noise,
    load_ocr_errors_dict,
    load_phonetic_errors_dict,
    two_d_array_choice,
    vectorized_choice,
)


def omit_rows(
    dataset_name: str,
    dataset_data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """
    Function that omits rows from a dataset and returns only the remaining rows.  Note that for the ACS and CPS datasets
      we need to account for oversampling in the PRL simulation so a helper function has been hadded here to do so.
    :param dataset_name: Dataset object being noised
    :param dataset_data:  pd.DataFrame of one of the dataset types used in Pseudopeople
    :param configuration: ConfigTree object containing noise level values
    :param randomness_stream: RandomnessStream object to make random selection for noise
    :return: pd.DataFrame with rows from the original dataframe removed
    """

    noise_level = configuration[Keys.ROW_PROBABILITY]
    # Omit rows
    to_noise_index = get_index_to_noise(
        dataset_data,
        noise_level,
        randomness_stream,
        f"{dataset_name}_omit_choice",
    )
    noised_data = dataset_data.loc[dataset_data.index.difference(to_noise_index)]

    return noised_data


def _get_census_omission_noise_levels(
    population: pd.DataFrame,
    base_probability: float = data_values.DO_NOT_RESPOND_BASE_PROBABILITY,
) -> pd.Series:
    """
    Helper function for do_not_respond noising based on demography of age, race/ethnicity, and sex.

    :param population: a dataset containing records of simulants
    :param base_probability: base probability for do_not_respond
    :return: a pd.Series of probabilities
    """
    probabilities = pd.Series(base_probability, index=population.index)
    probabilities += (
        population["race_ethnicity"]
        .astype(str)
        .map(data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_RACE)
    )
    for sex in ["Female", "Male"]:
        sex_mask = population["sex"] == sex
        age_bins = pd.cut(
            x=population[sex_mask]["age"],
            bins=data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE[sex].index,
        )
        probabilities[sex_mask] += age_bins.map(
            data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE[sex]
        ).astype(float)
    probabilities[probabilities < 0.0] = 0.0
    probabilities[probabilities > 1.0] = 1.0
    return probabilities


def apply_do_not_respond(
    dataset_name: str,
    dataset_data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """
    Applies targeted omission based on demographic model for census and surveys.

    :param dataset_name: Dataset object name being noised
    :param dataset_data:  pd.DataFrame of one of the form types used in Pseudopeople
    :param configuration: ConfigTree object containing noise level values
    :param randomness_stream: RandomnessStream object to make random selection for noise
    :return: pd.DataFrame with rows from the original dataframe removed
    """
    required_columns = ("age", "race_ethnicity", "sex")
    missing_columns = [col for col in required_columns if col not in dataset_data.columns]
    if len(missing_columns):
        raise ValueError(
            f"Dataset {dataset_name} is missing required columns: {missing_columns}"
        )

    # do_not_respond noise_levels are based on census
    noise_levels = _get_census_omission_noise_levels(dataset_data)

    # Apply an overall non-response rate of 27.6% for Current Population Survey (CPS)
    if dataset_name == DatasetNames.CPS:
        noise_levels += 0.276

    # Apply user-configured noise level
    configured_noise_level = configuration[Keys.ROW_PROBABILITY]
    default_noise_level = data_values.DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY[dataset_name]
    noise_levels = noise_levels * (configured_noise_level / default_noise_level)

    # Account for ACS and CPS oversampling
    if dataset_name in [DatasetNames.ACS, DatasetNames.CPS]:
        noise_levels = 0.5 + noise_levels / 2

    to_noise_idx = get_index_to_noise(
        dataset_data, noise_levels, randomness_stream, f"do_not_respond_{dataset_name}"
    )
    noised_data = dataset_data.loc[dataset_data.index.difference(to_noise_idx)]

    return noised_data


# def duplicate_rows(
#     dataset_data: pd.DataFrame,
#     configuration: ConfigTree,
#     randomness_stream: RandomnessStream,
# ) -> pd.DataFrame:
#     """

#     :param dataset_data:
#     :param configuration:
#     :param randomness_stream:
#     :return:
#     """
#     # todo actually duplicate rows
#     return dataset_data


def choose_wrong_options(
    data: pd.DataFrame,
    _: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that takes a categorical series and applies noise so some values has been replace with other options from
    a list.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :returns: pd.Series where data has been noised with other values from a list of possibilities
    """

    selection_type = {
        "employer_state": "state",
        "mailing_address_state": "state",
    }.get(str(column_name), column_name)

    selection_options = pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = selection_options.loc[selection_options[selection_type].notna(), selection_type]
    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(data),
        randomness_stream=randomness_stream,
        additional_key=f"{column_name}_incorrect_select_choice",
    ).to_numpy()

    return pd.Series(new_values, index=data.index, name=column_name)


def copy_from_household_member(
    data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :returns: pd.Series where data has been noised with other values from a list of possibilities
    """

    copy_values = data[COPY_HOUSEHOLD_MEMBER_COLS[column_name]]
    column = pd.Series(copy_values, index=data.index, name=column_name)
    return column


def swap_months_and_days(
    data: pd.DataFrame,
    _: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that swaps month and day of dates.

    :param data: A pandas dataframe containing necessary columns for column noise
    :param _: ConfigTree object containing noise level values
    :param randomness_stream: Randomness Stream object for random choices using vivarium CRN framework
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: Noised pd.Series where some dates have month and day swapped.
    """
    from pseudopeople.schema_entities import DATASETS, DATEFORMATS

    date_format = DATASETS.get_dataset(dataset_name).date_format

    column = data[column_name]
    if date_format == DATEFORMATS.YYYYMMDD:  # YYYYMMDD
        year = column.str[:4]
        month = column.str[4:6]
        day = column.str[6:]
        noised = year + day + month
    elif date_format == DATEFORMATS.MM_DD_YYYY:  # MM/DD/YYYY
        year = column.str[6:]
        month = column.str[:3]
        day = column.str[3:6]
        noised = day + month + year
    elif date_format == DATEFORMATS.MMDDYYYY:  # MMDDYYYY
        year = column.str[4:]
        month = column.str[:2]
        day = column.str[2:4]
        noised = day + month + year
    else:
        raise ValueError(f"Invalid date format in {dataset_name}.")

    return noised


def write_wrong_zipcode_digits(
    data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that noises a 5 digit zipcode

    :param data: A pandas dataframe containing necessary columns for column noise
    :param configuration:  Config tree object at column node.
    :param randomness_stream:  RandomnessStream object from Vivarium framework
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of noised zipcodes
    """

    column = data[column_name]
    str_len = column.str.len()
    if (str_len != 5).sum() > 0:
        # TODO: This is a BAD error message. It should never appear and if it
        #   does, the user shouldn't be checking the simulated population data.
        raise ValueError(
            "Zipcode data contains zipcodes that are not 5 digits long. Please check simulated population data."
        )

    rng = np.random.default_rng(
        get_hash(f"{randomness_stream.seed}_write_wrong_zipcode_digits")
    )
    shape = (len(column), 5)

    # todo: Update when vectorized choice is improved
    possible_replacements = list("0123456789")
    # Scale up noise levels to adjust for inclusive sampling with all numbers
    scaleup_factor = 1 / (1 - (1 / len(possible_replacements)))
    # Get configuration values for each piece of 5 digit zipcode
    digit_probabilities = scaleup_factor * np.array(
        configuration[Keys.ZIPCODE_DIGIT_PROBABILITIES]
    )
    replace = rng.random(shape) < digit_probabilities
    random_digits = rng.choice(possible_replacements, shape)
    digits = []
    for i in range(5):
        digit = np.where(replace[:, i], random_digits[:, i], column.str[i])
        digit = pd.Series(digit, index=column.index, name=column.name)
        digits.append(digit)

    new_zipcodes = digits[0] + digits[1] + digits[2] + digits[3] + digits[4]
    return new_zipcodes


def misreport_ages(
    data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """Function to mis-write ages based on perturbation parameters included in
    the config file.

    :param data: A pandas dataframe containing necessary columns for column noise
    :param configuration:  Config tree object at column node.
    :param randomness_stream:  RandomnessStream object from Vivarium framework
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series with some values noised from the original
    """

    column = data[column_name]
    possible_perturbations = configuration[Keys.POSSIBLE_AGE_DIFFERENCES].to_dict()
    perturbations = vectorized_choice(
        options=list(possible_perturbations.keys()),
        weights=list(possible_perturbations.values()),
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{column_name}_{column.name}_miswrite_ages",
    )
    new_values = column.astype(int) + perturbations
    # Reflect negative values to positive
    new_values[new_values < 0] *= -1
    # If new age == original age, subtract 1
    new_values[new_values == column.astype(int)] -= 1

    return new_values


def write_wrong_digits(
    data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that noises numeric characters in a series.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream

    returns: pd.Series with some numeric values experiencing noise.
    """
    column = data[column_name]
    if column.empty:
        return column
    # This is a fix to not replacing the original token for noise options
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY] / 0.9
    rng = np.random.default_rng(get_hash(f"{randomness_stream.seed}_write_wrong_digits"))
    column = column.astype(str)
    max_str_length = column.str.len().max()
    same_len_col = column.str.pad(max_str_length, side="right")
    is_number = pd.concat(
        [same_len_col.str[i].str.isdigit() for i in range(max_str_length)], axis=1
    )

    replace = (rng.random(is_number.shape) < token_noise_level) & is_number
    random_digits = rng.choice(list("0123456789"), is_number.shape)

    # Choose and replace values for a noised series
    noised_column = pd.Series("", index=column.index, name=column.name)
    digits = []
    for i in range(len(is_number.columns)):
        digit = np.where(replace.iloc[:, i], random_digits[:, i], same_len_col.str[i])
        digit = pd.Series(digit, index=column.index, name=column.name)
        digits.append(digit)
        noised_column = noised_column + digits[i]
    noised_column = noised_column.str.strip()

    return noised_column


def use_nicknames(
    data: pd.DataFrame,
    _: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that replaces a name with a choice of potential nicknames.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of nicknames replacing original names
    """
    nicknames = load_nicknames_data()
    column = data[column_name]
    have_nickname_idx = column.index[column.isin(nicknames.index)]
    noised = two_d_array_choice(
        column.loc[have_nickname_idx], nicknames, randomness_stream, column_name
    )
    column.loc[have_nickname_idx] = noised
    return column


def use_fake_names(
    data: pd.DataFrame,
    _: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return:
    """
    column = data[column_name]
    fake_first = fake_first_names
    fake_last = fake_last_names
    fake_names = {
        "first_name": fake_first,
        "middle_name": fake_first,
        "last_name": fake_last,
        "spouse_first_name": fake_first,
        "spouse_last_name": fake_last,
        "dependent_1_first_name": fake_first,
        "dependent_1_last_name": fake_last,
        "dependent_2_first_name": fake_first,
        "dependent_2_last_name": fake_last,
        "dependent_3_first_name": fake_first,
        "dependent_3_last_name": fake_last,
        "dependent_4_first_name": fake_first,
        "dependent_4_last_name": fake_last,
    }
    options = fake_names[column_name]

    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{column_name}_fake_names",
    )
    return pd.Series(new_values, index=column.index, name=column.name)


def make_phonetic_errors(
    data: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: Any,
) -> pd.Series:
    """

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of noised data
    """

    phonetic_error_dict = load_phonetic_errors_dict()

    def phonetic_corrupt(truth, corrupted_pr, rng):
        err = ""
        i = 0
        while i < len(truth):
            error_introduced = False
            for token_length in [7, 6, 5, 4, 3, 2, 1]:
                token = truth[i : (i + token_length)]
                if token in phonetic_error_dict and not error_introduced:
                    if rng.uniform() < corrupted_pr:
                        err += rng.choice(phonetic_error_dict[token])
                        i += token_length
                        error_introduced = True
            if not error_introduced:
                err += truth[i : (i + 1)]
                i += 1
        return err

    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    rng = np.random.default_rng(
        seed=get_hash(f"{randomness_stream.seed}_make_phonetic_errors")
    )
    column = data[column_name]
    column = column.astype(str)
    for idx in column.index:
        noised_value = phonetic_corrupt(
            column[idx],
            token_noise_level,
            rng,
        )
        column[idx] = noised_value
    return column


def leave_blanks(
    data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that takes a column and blanks out all values.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    """
    return pd.Series(np.nan, index=data.index, name=column_name)


def make_typos(
    data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """Function that applies noise to the string values
    representative of keyboard mistyping.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :returns: pd.Series of column with noised data
    """
    with open(paths.QWERTY_ERRORS) as f:
        qwerty_errors = yaml.safe_load(f)
    qwerty_errors = pd.DataFrame.from_dict(qwerty_errors, orient="index")
    column = data[column_name]
    if column.empty:
        return column
    column = column.astype(str)
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    # TODO: remove this hard-coding
    include_token_probability_level = 0.1
    rng = np.random.default_rng(seed=get_hash(f"{randomness_stream.seed}_make_typos"))

    # Make all strings the same length by padding with spaces
    max_str_length = column.str.len().max()
    same_len_col = column.str.pad(max_str_length, side="right")

    is_typo_option = pd.concat(
        [same_len_col.str[i].isin(qwerty_errors.index) for i in range(max_str_length)], axis=1
    )
    replace = is_typo_option & (rng.random(is_typo_option.shape) < token_noise_level)
    keep_original = replace & (
        rng.random(is_typo_option.shape) < include_token_probability_level
    )

    # Loop through each column of string elements and apply noising
    noised_column = pd.Series("", index=column.index, name=column.name)
    for i in range(max_str_length):
        orig = same_len_col.str[i]
        replace_mask = replace.iloc[:, i]
        keep_original_mask = keep_original.iloc[:, i]
        typos = two_d_array_choice(
            data=orig[replace_mask],
            options=qwerty_errors,
            randomness_stream=randomness_stream,
            additional_key=f"{column_name}_{i}",
        )
        characters = orig.copy()
        characters[replace_mask] = typos
        characters[keep_original_mask] = characters + orig
        noised_column = noised_column + characters
    noised_column = noised_column.str.strip()

    return noised_column


def make_ocr_errors(
    data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of noised data
    """

    # Load OCR error dict
    ocr_error_dict = load_ocr_errors_dict()

    def ocr_corrupt(truth, corrupted_pr, rng):
        err = ""
        i = 0
        while i < len(truth):
            error_introduced = False
            for token_length in [3, 2, 1]:
                token = truth[i : (i + token_length)]
                if token in ocr_error_dict and not error_introduced:
                    if rng.uniform() < corrupted_pr:
                        err += rng.choice(ocr_error_dict[token])
                        i += token_length
                        error_introduced = True
                        break
            if not error_introduced:
                err += truth[i : (i + 1)]
                i += 1
        return err

    # Apply keyboard corrupt for OCR to column
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    rng = np.random.default_rng(seed=get_hash(f"{randomness_stream.seed}_make_ocr_errors"))
    column = data[column_name]
    column = column.astype(str)
    for idx in column.index:
        noised_value = ocr_corrupt(
            column.loc[idx],
            token_noise_level,
            rng,
        )
        column[idx] = noised_value

    return column
