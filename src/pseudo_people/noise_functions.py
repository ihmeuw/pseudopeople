import pandas as pd

from pseudo_people.configuration import (
    ColumnNoiseConfigurationNode,
    RowNoiseConfigurationNode,
)


def omit_rows(
    form_data: pd.DataFrame, configuration: RowNoiseConfigurationNode
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :return:
    """
    # todo actually omit rows
    return form_data


def duplicate_rows(
    form_data: pd.DataFrame, configuration: RowNoiseConfigurationNode
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :return:
    """
    # todo actually duplicate rows
    return form_data


def generate_nicknames(
    column: pd.Series, configuration: ColumnNoiseConfigurationNode
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :return:
    """
    # todo actually generate nicknames
    return column


def generate_fake_names(
    column: pd.Series, configuration: ColumnNoiseConfigurationNode
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :return:
    """
    # todo actually generate fake names
    return column


def generate_phonetic_errors(
    column: pd.Series, configuration: ColumnNoiseConfigurationNode
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :return:
    """
    # todo actually generate fake names
    return column


# todo add noise functions
