from typing import Any

import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.constants import paths
from pseudopeople.utilities import vectorized_choice


def omit_rows(
    form_data: pd.DataFrame,
    configuration: float,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :return:
    """
    # todo actually omit rows
    return form_data


def duplicate_rows(
    form_data: pd.DataFrame,
    configuration: float,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :return:
    """
    # todo actually duplicate rows
    return form_data


def generate_incorrect_selections(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """
    Function that takes a categorical series and applies noise so some values has been replace with other options from
    a list.

    :param column:  A categorical pd.Series
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param additional_key: Key for RandomnessStream
    :returns: pd.Series where data has been noised with other values from a list of possibilities
    """

    column = column.copy()
    col = column.name
    incorrect_selections = pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = incorrect_selections.loc[incorrect_selections[col].notna(), col]
    noise_level = configuration.row_noise_level

    # Select indices to noise and noise data
    to_noise_idx = randomness_stream.filter_for_probability(
        column.index,
        probability=noise_level,
        additional_key=f"{additional_key}_{col}_incorrect_select_filter",
    )
    column[to_noise_idx] = vectorized_choice(
        options=options,
        n_to_choose=len(to_noise_idx),
        randomness_stream=randomness_stream,
        additional_key=f"{additional_key}_{col}_incorrect_select_choice",
    )

    return column


def generate_within_household_copies(
    form_data: pd.DataFrame,
    configuration: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return form_data


def swap_months_and_days(
    form_data: pd.DataFrame,
    configuration: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return form_data


def miswrite_zip_codes(
    form_data: pd.DataFrame,
    configuration: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return form_data


def miswrite_ages(
    form_data: pd.DataFrame,
    configuration: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return form_data


def miswrite_numerics(
    form_data: pd.DataFrame,
    configuration: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return form_data


def generate_nicknames(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate nicknames
    return column


def generate_fake_names(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate fake names
    return column


def generate_phonetic_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate fake names
    return column


def generate_missing_data(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """
    Function that takes a column and blanks out a configurable portion of its data to be missing.

    :param column:  pd.Series of data
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param additional_key: Key for RandomnessStream
    :returns: pd.Series of column with configured amount of data missing as an empty string.
    """

    # Avoid SettingWithCopyWarning
    column = column.copy()
    noise_level = configuration.row_noise_level
    # Get rows to noise
    to_noise_idx = randomness_stream.filter_for_probability(
        column.index,
        probability=noise_level,
        additional_key=f"{additional_key}_missing_data_filter",
    )
    column.loc[to_noise_idx] = ""

    return column


def generate_typographical_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate typographical errors
    return column


def generate_ocr_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate OCR errors
    return column


# todo add noise functions
