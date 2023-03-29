from typing import Any

import numpy as np
import pandas as pd
import yaml
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.constants import paths


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


def miswrite_zipcodes(
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
    to_noise_idx = _get_to_noise_idx(
        column,
        configuration,
        randomness_stream,
        additional_key,
        context_key="missing_data_filter",
    )
    column.loc[to_noise_idx] = ""

    return column


def generate_typographical_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """Function that takes a column and applies noise to the string values
    representative of keyboard mis-typing.

    :param column:  pd.Series of data
    :param configuration: ConfigTree object containing noising parameters
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN
    :param additional_key: Key for RandomnessStream
    :returns: pd.Series of column with noised data
    """
    column = column.copy()
    not_missing_idx = column.index[(column.notna()) & (column != "")]

    with open(paths.QWERTY_ERRORS) as f:
        qwerty_errors = yaml.full_load(f)

    def keyboard_corrupt(truth, corrupted_pr, addl_pr, rng):
        """Abie's implementation of typographical noising"""
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

    token_noise_level = configuration.token_noise_level
    include_original_token_level = configuration.include_original_token_level

    to_noise_idx = _get_to_noise_idx(
        column.loc[not_missing_idx],
        configuration,
        randomness_stream,
        additional_key,
        context_key="typographical_noise_filter",
    )
    rng = np.random.default_rng(seed=randomness_stream.seed)
    for idx in to_noise_idx:
        noised_value = keyboard_corrupt(
            column[idx],
            token_noise_level,
            include_original_token_level,
            rng,
        )
        column[idx] = noised_value

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


####################
# HELPER FUNCTIONS #
####################
def _get_to_noise_idx(column, configuration, randomness_stream, additional_key, context_key):
    noise_level = configuration.row_noise_level
    # Get rows to noise
    to_noise_idx = randomness_stream.filter_for_probability(
        column.index,
        probability=noise_level,
        additional_key=f"{additional_key}_{context_key}",
    )

    return to_noise_idx
