from typing import Any

import numpy as np
import pandas as pd
import yaml
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.configuration import Keys
from pseudopeople.constants import data_values, paths
from pseudopeople.constants.metadata import Attributes, DatasetNames
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.exceptions import ConfigurationError
from pseudopeople.noise_scaling import load_nicknames_data
from pseudopeople.utilities import (
    get_index_to_noise,
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


def generate_incorrect_selections(
    column: pd.Series,
    _: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """
    Function that takes a categorical series and applies noise so some values has been replace with other options from
    a list.

    :param column:  A categorical pd.Series
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param additional_key: Key for RandomnessStream
    :returns: pd.Series where data has been noised with other values from a list of possibilities
    """

    selection_type = {
        "employer_state": "state",
        "mailing_address_state": "state",
    }.get(str(column.name), column.name)

    selection_options = pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = selection_options.loc[selection_options[selection_type].notna(), selection_type]
    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{additional_key}_incorrect_select_choice",
    ).to_numpy()

    return pd.Series(new_values, index=column.index, name=column.name)


# def generate_within_household_copies(
#     column: pd.Series,
#     configuration: ConfigTree,
#     randomness_stream: RandomnessStream,
#     additional_key: Any,
# ) -> pd.Series:
#     """

#     :param column:
#     :param configuration:
#     :param randomness_stream:
#     :param additional_key: Key for RandomnessStream
#     :return:
#     """
#     # todo actually duplicate rows
#     return column


def swap_months_and_days(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """
    Function that swaps month and day of dates.

    :param column: pd.Series containing dates with the format YYYY-MM-DD
    :param configuration: ConfigTree object containing noise level values
    :param randomness_stream: Randomness Stream object for random choices using vivarium CRN framework
    :param additional_key: Key for RandomnessStream
    :return: Noised pd.Series where some dates have month and day swapped.
    """
    from pseudopeople.schema_entities import COLUMNS, DATEFORMATS

    column_type = COLUMNS.get_column(column.name)
    try:
        date_format = column_type.additional_attributes[Attributes.DATE_FORMAT]
    except KeyError:
        raise ConfigurationError(
            f"Error while running noise function `swap_months_and_days' on column '{column.name}'. "
            f"'{column.name}' does not have attribute date format. "
        )

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
    else:
        raise ValueError(
            f"Invalid datetime format in {column.name}.  Please check input data."
        )

    return noised


def miswrite_zipcodes(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    _: Any,
) -> pd.Series:
    """
    Function that noises a 5 digit zipcode

    :param column: A pd.Series of 5 digit zipcodes as strings
    :param configuration:  Config tree object at column node.
    :param randomness_stream:  RandomnessStream object from Vivarium framework
    :return: pd.Series of noised zipcodes
    """

    str_len = column.str.len()
    if (str_len != 5).sum() > 0:
        raise ValueError(
            "Zipcode data contains zipcodes that are not 5 digits long. Please check input data."
        )

    rng = np.random.default_rng(randomness_stream.seed)
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


def miswrite_ages(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """Function to mis-write ages based on perturbation parameters included in
    the config file.

    :param column: pd.Series of ages
    :param configuration: ConfigTree
    :param randomness_stream: Vivarium RandomnessStream
    :param additional_key: additional key used for randomness_stream calls
    :return:
    """
    possible_perturbations = configuration[Keys.POSSIBLE_AGE_DIFFERENCES].to_dict()
    perturbations = vectorized_choice(
        options=list(possible_perturbations.keys()),
        weights=list(possible_perturbations.values()),
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{additional_key}_{column.name}_miswrite_ages",
    )
    new_values = column.astype(int) + perturbations
    # Reflect negative values to positive
    new_values[new_values < 0] *= -1
    # If new age == original age, subtract 1
    new_values[new_values == column.astype(int)] -= 1

    return new_values


def miswrite_numerics(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    _: Any,
) -> pd.Series:
    """
    Function that noises numeric characters in a series.

    :param column: A pd.Series
    :param configuration: ConfigTree object containing noise level
    :param randomness_stream: RandomnessStream for CRN framework.

    returns: pd.Series with some numeric values experiencing noise.
    """
    if column.empty:
        return column
    # This is a fix to not replacing the original token for noise options
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY] / 0.9
    rng = np.random.default_rng(randomness_stream.seed)
    column = column.astype(str)
    longest_str = column.str.len().max()
    same_len_col = column.str.pad(longest_str, side="right")
    is_number = pd.concat(
        [same_len_col.str[i].str.isdigit() for i in range(longest_str)], axis=1
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
    noised_column.str.strip()

    return noised_column


def generate_nicknames(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """
    Function that replaces a name with a choice of potential nicknames.

    :param column: pd.Series of names
    :param configuration: ConfigTree object containing noise level values
    :param randomness_stream: RandomnessStream object to use vivarium CRN framework.
    :param additional_key: Key for RandomnessStream
    :return: pd.Series of nicknames replacing original names
    """
    nicknames = load_nicknames_data()

    have_nickname_idx = column.index[column.isin(nicknames.index)]
    noised = two_d_array_choice(
        column.loc[have_nickname_idx], nicknames, randomness_stream, additional_key
    )
    column.loc[have_nickname_idx] = noised
    return column


def generate_fake_names(
    column: pd.Series,
    _: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column: pd.Series of names
    :param _:  ConfigTree object with noise level values
    :param randomness_stream:  RandomnessStream instance of vivarium
    :param additional_key: Key for RandomnessStream
    :return:
    """
    name = column.name
    fake_first = fake_first_names
    fake_last = fake_last_names
    fake_names = {"first_name": fake_first, "last_name": fake_last}
    options = fake_names[name]

    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{additional_key}_fake_names",
    )
    return pd.Series(new_values, index=column.index, name=column.name)


# def generate_phonetic_errors(
#     column: pd.Series,
#     configuration: ConfigTree,
#     randomness_stream: RandomnessStream,
#     additional_key: Any,
# ) -> pd.Series:
#     """

#     :param column:
#     :param configuration:
#     :param randomness_stream:
#     :param additional_key: Key for RandomnessStream
#     :return:
#     """
#     # todo actually generate fake names
#     return column


def generate_missing_data(column: pd.Series, *_: Any) -> pd.Series:
    """
    Function that takes a column and blanks out all values.

    :param column:  pd.Series of data
    :returns: pd.Series of empty strings with the index of column.
    """

    return pd.Series(np.nan, index=column.index)


def generate_typographical_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    _: Any,
) -> pd.Series:
    """Function that takes a column and applies noise to the string values
    representative of keyboard mistyping.

    :param column:  pd.Series of data
    :param configuration: ConfigTree object containing noising parameters
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN
    :returns: pd.Series of column with noised data
    """

    with open(paths.QWERTY_ERRORS) as f:
        qwerty_errors = yaml.full_load(f)

    def keyboard_corrupt(truth, corrupted_pr, addl_pr, rng):
        """For each string, loop through each character and determine if
        it is to be corrupted. If so, uniformly choose from the appropriate
        values to mistype. Also determine which mistyped characters should
        include the original value and, if it does, include the original value
        after the mistyped value
        """
        err = ""
        i = 0
        while i < len(truth):
            error_introduced = False
            token = truth[i : (i + 1)]
            if token in qwerty_errors and not error_introduced:
                random_number = rng.uniform()
                if random_number < corrupted_pr:
                    err += rng.choice(qwerty_errors[token])
                    random_number = rng.uniform()
                    if random_number < addl_pr:
                        err += token
                    i += 1
                    error_introduced = True
            if not error_introduced:
                err += truth[i : (i + 1)]
                i += 1
        return err

    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    # TODO: remove this hard-coding
    include_token_probability_level = 0.1

    rng = np.random.default_rng(seed=randomness_stream.seed)
    column = column.astype(str)
    for idx in column.index:
        noised_value = keyboard_corrupt(
            column[idx],
            token_noise_level,
            include_token_probability_level,
            rng,
        )
        column[idx] = noised_value

    return column


# def generate_ocr_errors(
#     column: pd.Series,
#     configuration: ConfigTree,
#     randomness_stream: RandomnessStream,
#     additional_key: Any,
# ) -> pd.Series:
#     """

#     :param column:
#     :param configuration:
#     :param randomness_stream:
#     :param additional_key: Key for RandomnessStream
#     :return:
#     """
#     # todo actually generate OCR errors
#     return column
