import sys
from typing import Any, Union

import numpy as np
import pandas as pd
from loguru import logger
from vivarium.framework.randomness import RandomnessStream, random

from pseudopeople.constants import paths


def get_randomness_stream(dataset_name: str, seed: int) -> RandomnessStream:
    return RandomnessStream(dataset_name, lambda: pd.Timestamp("2020-04-01"), seed)


def vectorized_choice(
    options: Union[list, pd.Series],
    n_to_choose: int,
    randomness_stream: RandomnessStream = None,
    weights: Union[list, pd.Series] = None,
    additional_key: Any = None,
    random_seed: int = None,
):
    """
    Function that takes a list of options and uses Vivarium common random numbers framework to make a given number
    of random choice selections.

    :param options: List and series of possible values to choose
    :param n_to_choose: Number of choices to make, the length of the returned array of values
    :param randomness_stream: RandomnessStream being used for Vivarium's CRN framework
    :param weights: List or series containing weights for each options
    :param additional_key: Key to pass to randomness_stream
    :param random_seed: Seed to pass to randomness_stream.
    Note additional_key and random_seed are used to make calls using a RandomnessStream unique

    returns: ndarray
    """
    if not randomness_stream and (additional_key == None and random_seed == None):
        raise RuntimeError(
            "An additional_key and a random_seed are required in 'vectorized_choice'"
            + "if no RandomnessStream is passed in"
        )
    if weights is None:
        n = len(options)
        weights = np.ones(n) / n
    if isinstance(weights, list):
        weights = np.array(weights)
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


def get_index_to_noise(
    data: Union[pd.DataFrame, pd.Series],
    noise_level: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Index:
    """
    Function that takes a series and returns a pd.Index that chosen by Vivarium Common Random Number to be noised.
    """

    # Get rows to noise
    if isinstance(data, pd.Series):
        not_empty_idx = data.index[(data != "") & (data.notna())]
    else:
        not_empty_idx = data.index
    to_noise_idx = randomness_stream.filter_for_probability(
        not_empty_idx,
        probability=noise_level,
        additional_key=additional_key,
    )

    return to_noise_idx


def noise_scaling_incorrect_selection(name: str) -> float:
    """
    Function to scale noising for incorrect selection to adjust for the possibility of noising with the original values.
    """
    selection_type = {
        "employer_state": "state",
        "mailing_address_state": "state",
    }.get(name, name)

    selection_options = pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)
    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = selection_options.loc[selection_options[selection_type].notna(), selection_type]

    # Scale to adjust for possibility of noising with original value
    noise_scaling_value = 1 / (1 - 1 / len(options))

    return noise_scaling_value


def configure_logging_to_terminal(verbose: bool = False):
    logger.remove()  # Clear default configuration
    add_logging_sink(sys.stdout, verbose, colorize=True)


def add_logging_sink(sink, verbose, colorize=False, serialize=False):
    message_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )
    if verbose:
        logger.add(
            sink, colorize=colorize, level="DEBUG", format=message_format, serialize=serialize
        )
    else:
        logger.add(
            sink, colorize=colorize, level="INFO", format=message_format, serialize=serialize
        )
