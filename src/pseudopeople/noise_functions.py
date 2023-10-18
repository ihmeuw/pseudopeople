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
from pseudopeople.noise_scaling import (
    load_incorrect_select_options,
    load_nicknames_data,
)
from pseudopeople.utilities import (
    get_index_to_noise,
    load_ocr_errors_dict,
    load_phonetic_errors_dict,
    load_qwerty_errors_data,
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
    ages = pd.Series(np.arange(population["age"].max() + 1))
    for sex in ["Female", "Male"]:
        effect_by_age_bin = data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE[sex]
        # NOTE: calling pd.cut on a large array with an IntervalIndex is slow,
        # see https://github.com/pandas-dev/pandas/issues/47614
        # Instead, we only pd.cut the unique ages, then do a simpler `.map` on the age column
        age_bins = pd.cut(ages, bins=effect_by_age_bin.index)
        effect_by_age = pd.Series(
            age_bins.map(effect_by_age_bin),
            index=ages,
        )
        sex_mask = population["sex"] == sex
        probabilities[sex_mask] += (
            population[sex_mask]["age"].map(effect_by_age).astype(float)
        )
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

    selection_options = load_incorrect_select_options()

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
    possible_replacements = np.array(list("0123456789"))
    # Scale up noise levels to adjust for inclusive sampling with all numbers
    scaleup_factor = 1 / (1 - (1 / len(possible_replacements)))
    # Get configuration values for each piece of 5 digit zipcode
    digit_probabilities = scaleup_factor * np.array(
        configuration[Keys.ZIPCODE_DIGIT_PROBABILITIES]
    )
    replace = rng.random(shape) < digit_probabilities
    num_to_replace = replace.sum()
    random_digits = rng.choice(possible_replacements, num_to_replace)

    # https://stackoverflow.com/a/9493192/
    # Changing this to a U5 numpy string type means that each string will have exactly 5 characters.
    # view("U1") then reinterprets this memory as an array of individual (Unicode) characters.
    same_len_col_exploded = column.values.astype("U5").view("U1").reshape(shape)
    same_len_col_exploded[replace] = random_digits
    return pd.Series(
        same_len_col_exploded.view("U5").reshape(len(column)),
        index=column.index,
        name=column.name,
    )


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

    possible_replacements = np.array(list("0123456789"))

    # https://stackoverflow.com/a/9493192/
    # Changing this to a numpy (not Python) string type means that it will have a fixed
    # number of characters, equal to the longest string in the array.
    # view("U1") then reinterprets this memory as an array of individual (Unicode) characters.
    same_len_col_exploded = (
        column.values.astype(str).view("U1").reshape((len(column), max_str_length))
    )
    # Surprisingly, Numpy does not provide a computationally efficient way to do
    # this check for which characters are eligible.
    # A head-to-head comparison found np.isin to be orders of magnitude slower than using Pandas here.
    # Also, np.isin fails silently with sets (see https://numpy.org/doc/stable/reference/generated/numpy.isin.html)
    # so be careful if testing that in the future!
    is_number = (
        pd.DataFrame(same_len_col_exploded).isin(set(possible_replacements)).to_numpy()
    )

    replace = np.zeros_like(is_number, dtype=bool)
    replace[is_number] = rng.random(is_number.sum()) < token_noise_level
    num_to_replace = replace.sum()
    random_digits = rng.choice(possible_replacements, num_to_replace)

    same_len_col_exploded[replace] = random_digits
    noised_column = same_len_col_exploded.view(f"U{max_str_length}").reshape(len(column))

    return pd.Series(noised_column, index=column.index, name=column.name)


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
    nickname_eligible_names = set(nicknames.index)
    column = data[column_name]
    have_nickname_idx = column.index[column.isin(nickname_eligible_names)]
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
                        break
            if not error_introduced:
                err += truth[i : (i + 1)]
                i += 1
        return err

    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    rng = np.random.default_rng(
        seed=get_hash(f"{randomness_stream.seed}_make_phonetic_errors")
    )
    return (
        data[column_name]
        .astype(str)
        .apply(phonetic_corrupt, corrupted_pr=token_noise_level, rng=rng)
    )


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

    qwerty_errors = load_qwerty_errors_data()
    qwerty_errors_eligible_chars = set(qwerty_errors.index)

    column = data[column_name]
    if column.empty:
        return column
    column = column.astype(str)
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    # TODO: remove this hard-coding
    include_token_probability_level = 0.1
    rng = np.random.default_rng(seed=get_hash(f"{randomness_stream.seed}_make_typos"))

    same_len_col_exploded = (
        # Somewhat counterintuitively, .astype(str) turns the column into a numpy,
        # fixed-length string type, U#, where # is the length of the longest string.
        column.values.astype(str)
        # Split into individual characters
        .view("U1").reshape((len(column), -1))
        # https://stackoverflow.com/a/9493192/
        # NOTE: Surprisingly, casting this to "object" (Python string type) seems to make
        # the rest of this method faster. Without this cast, you have to use np.char.add to concatenate,
        # which seems very slow for some reason?
        .astype("object")
    )

    # Surprisingly, Numpy does not provide a computationally efficient way to do
    # this check for which characters are eligible.
    # A head-to-head comparison found np.isin to be orders of magnitude slower than using Pandas here.
    # Also, np.isin fails silently with sets (see https://numpy.org/doc/stable/reference/generated/numpy.isin.html)
    # so be careful if testing that in the future!
    is_typo_option = (
        pd.DataFrame(same_len_col_exploded).isin(qwerty_errors_eligible_chars).to_numpy()
    )
    replace = np.zeros_like(is_typo_option)
    replace[is_typo_option] = rng.random(is_typo_option.sum()) < token_noise_level
    keep_original = np.zeros_like(replace)
    keep_original[replace] = rng.random(replace.sum()) < include_token_probability_level

    # Apply noising
    to_replace = same_len_col_exploded[replace]
    replace_random = rng.random(replace.sum())
    number_of_options = qwerty_errors.count(axis=1)
    replace_option_index = np.floor(
        replace_random * number_of_options.loc[to_replace].to_numpy()
    )
    originals_to_keep = same_len_col_exploded[keep_original]
    # Numpy does not have an efficient way to do this merge operation
    same_len_col_exploded[replace] = (
        pd.DataFrame({"to_replace": to_replace, "replace_option_index": replace_option_index})
        .reset_index()
        .merge(
            qwerty_errors.stack().rename("replacement"),
            left_on=["to_replace", "replace_option_index"],
            right_index=True,
        )
        # Merge does not return the results in order
        .sort_values("index")
        .replacement.to_numpy()
    )

    same_len_col_exploded[keep_original] += originals_to_keep

    noised_column = np.sum(same_len_col_exploded, axis=1)

    return pd.Series(noised_column, index=column.index, name=column.name)


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

    return (
        data[column_name]
        .astype(str)
        .apply(ocr_corrupt, corrupted_pr=token_noise_level, rng=rng)
    )
