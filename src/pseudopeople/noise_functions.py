import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import filter_by_rate, RandomnessStream

from pseudopeople.utilities import (
    vectorized_choice,
)
from pseudopeople.constants import paths


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
    # Load and format nicknames dataset
    nicknames = pd.read_csv(paths.NICKNAMES_DATA, header=None)
    nicknames = nicknames.apply(lambda x: x.astype(str).str.title()).set_index(0)

    # Find individuals eligible to use nicknames
    eligible_for_noise_idx = column.index[column.isin(nicknames.index)]
    l_names = len(column.loc[eligible_for_noise_idx].unique())
    # TODO: replace with configuration
    p = 0.5
    p_name = 0.5 / l_names

    # Cycle through unique list of names and pick which nickname to use
    # TODO: Import vectorized choice
    # Take length of unique values of column and noise that level for each name
    for name in column.loc[eligible_for_noise_idx].unique():
        sims_to_noise_idx = randomness_stream.filter_by_rate(
            column.index[column == name],
            rate=list(p_name),
            additional_key=f"{name}_noise_filter",
        )
        column.loc[sims_to_noise_idx] = vectorized_choice(
            options=nicknames.loc[nicknames.index == name].values,
            n_to_choose=len(sims_to_noise_idx),
            randomness_stream=randomness_stream,
            additional_key=f"{name}_nickname",
        ).to_numpy()

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
