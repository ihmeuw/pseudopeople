import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.utilities import (
    filter_by_rate,
    vectorized_choicem,
)


def omit_rows(
    form_data: pd.DataFrame, configuration: float, randomness_stream: RandomnessStream
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
    form_data: pd.DataFrame, configuration: float, randomness_stream: RandomnessStream
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :return:
    """
    # todo actually duplicate rows
    return form_data


def generate_nicknames(
    column: pd.Series, configuration: ConfigTree, randomness_stream: RandomnessStream
) -> pd.Series:
    """
    Function to noise eligible names so "nicknames" are used in forms instead of an individual's "real" name.

    :param column:  Column containing names to be noised with alternative nicknames
    :param configuration:  ConfigTree object containing level at which to noise column
    :param randomness_stream:  RandomnessStream object to utilize Vivarium common random numbers.

    :return:
    Series containing names that have been noised at the provided level.
    """
    # todo actually generate nicknames
    return column


def generate_fake_names(
    column: pd.Series, configuration: ConfigTree, randomness_stream: RandomnessStream
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :return:
    """
    # todo actually generate fake names
    return column


def generate_phonetic_errors(
    column: pd.Series, configuration: ConfigTree, randomness_stream: RandomnessStream
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :return:
    """
    # todo actually generate fake names
    return column


# todo add noise functions
