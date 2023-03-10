from typing import Union, Any

import numpy as np
import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream, random

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
    nicknames = pd.read_csv(paths.NICKNAMES_DATA, header=None, keep_default_na=False)
    nicknames = nicknames.apply(lambda x: x.astype(str).str.title()).set_index(0)

    # Find individuals eligible to use nicknames
    # TODO: fix for configuration
    noise_level = configuration.row_noise_level
    eligible_for_noise_idx = column.index[column.isin(nicknames.index)]
    nicknames_idx = randomness_stream.filter_for_probability(
        eligible_for_noise_idx,
        probability=noise_level,
        additional_key="nickname_noise_filter",
    )

    # Cycle through all possible nicknames and pick simulants who use a nickname.
    # TODO: update vectorized choice to be completely vectorized
    for name in column.loc[nicknames_idx].unique():
        name_idx = column.index[column == name]
        to_noise_idx = nicknames_idx.intersection(name_idx)
        column.loc[to_noise_idx] = vectorized_choice(
            options=nicknames.loc[name].values,
            n_to_choose=len(to_noise_idx),
            randomness_stream=randomness_stream,
            additional_key=f"{name}_nickname",
        )

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


def vectorized_choice(
    options: Union[pd.Series, np.array],
    n_to_choose: int,
    randomness_stream: RandomnessStream = None,
    weights: Union[pd.Series, np.array] = None,
    additional_key: Any = None,
    random_seed: int = None,
):
    if not randomness_stream and (additional_key == None and random_seed == None):
        raise RuntimeError(
            "An additional_key and a random_seed are required in 'vectorized_choice'"
            + "if no RandomnessStream is passed in"
        )
    if weights is None:
        n = len(options)
        weights = np.ones(n) / n
    # for each of n_to_choose, sample uniformly between 0 and 1
    index = pd.Index(np.arange(n_to_choose))
    if randomness_stream is None:
        # Generate an additional_key on-the-fly and use that in randomness.random
        additional_key = f"{additional_key}_{random_seed}"
        probs = random(str(additional_key), index)
    else:
        probs = randomness_stream.get_draw(index, additional_key=additional_key)

    # build cdf based on weights
    pmf = weights / weights.sum()
    cdf = np.cumsum(pmf)

    # for each p_i in probs, count how many elements of cdf for which p_i >= cdf_i
    chosen_indices = np.searchsorted(cdf, probs, side="right")
    return np.take(options, chosen_indices)