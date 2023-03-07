import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream


def omit_rows(
    form_data: pd.DataFrame, configuration: float, randomness_stream: RandomnessStream
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
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

    :param column:
    :param configuration:
    :param randomness_stream:
    :return:
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
