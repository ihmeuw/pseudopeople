import sys
from functools import cache
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from vivarium.framework.randomness import RandomnessStream, get_hash
from vivarium.framework.randomness.index_map import IndexMap

from pseudopeople.constants import metadata, paths


def get_randomness_stream(dataset_name: str, seed: Any, index: pd.Index) -> RandomnessStream:
    map_size = max(1_000_000, max(index) * 2)
    return RandomnessStream(
        key=dataset_name,
        clock=lambda: pd.Timestamp("2020-04-01"),
        seed=seed,
        index_map=IndexMap(size=map_size),
    )


def vectorized_choice(
    options: Union[list, pd.Series],
    n_to_choose: int,
    randomness_stream: RandomnessStream,
    weights: Union[list, pd.Series] = None,
    additional_key: Any = None,
) -> pd.Series:
    """
    Function that takes a list of options and uses Vivarium common random numbers framework to make a given number
    of random choice selections.

    :param options: List and series of possible values to choose
    :param n_to_choose: Number of choices to make, the length of the returned array of values
    :param randomness_stream: RandomnessStream being used for Vivarium's CRN framework
    :param weights: List or series containing weights for each options
    :param additional_key: Key to pass to randomness_stream

    returns: ndarray
    """
    # for each of n_to_choose, sample uniformly between 0 and 1
    index = pd.Index(np.arange(n_to_choose))
    probs = randomness_stream.get_draw(index, additional_key=additional_key)

    if weights is None:
        chosen_indices = np.floor(probs * len(options)).astype(int)
    else:
        if isinstance(weights, list):
            weights = np.array(weights)
        # build cdf based on weights
        pmf = weights / weights.sum()
        cdf = np.cumsum(pmf)

        # for each p_i in probs, count how many elements of cdf for which p_i >= cdf_i
        chosen_indices = np.searchsorted(cdf, probs, side="right")

    return np.take(options, chosen_indices, axis=0)


def get_index_to_noise(
    data: pd.DataFrame,
    noise_level: Union[float, pd.Series],
    randomness_stream: RandomnessStream,
    additional_key: Any,
    is_column_noise: bool = False,
    missingness: Optional[pd.DataFrame] = None,
) -> pd.Index:
    """
    Function that takes a series and returns a pd.Index that chosen by Vivarium Common Random Number to be noised.
    """

    # Get rows to noise
    if is_column_noise:
        if missingness is None:
            missingness = data.isna() | (data == "")
        missing = missingness.any(axis=1)
        eligible_for_noise_idx = data.index[~missing]
    else:
        # Any index can be noised for row noise
        eligible_for_noise_idx = data.index

    # As long as noise is relatively rare, it will be faster to randomly select cells to
    # noise rather than generating a random draw for every item eligible
    if isinstance(noise_level, float) and noise_level < 0.2:
        rng = np.random.default_rng(
            seed=get_hash(f"{randomness_stream.seed}_get_index_to_noise_{additional_key}")
        )
        number_to_noise = rng.binomial(len(eligible_for_noise_idx), p=noise_level)
        to_noise_idx = pd.Index(
            rng.choice(eligible_for_noise_idx, size=number_to_noise, replace=False)
        )
    else:
        to_noise_idx = randomness_stream.filter_for_probability(
            eligible_for_noise_idx,
            probability=noise_level,
            additional_key=additional_key,
        )

    return to_noise_idx


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


def two_d_array_choice(
    data: pd.Series,
    options: pd.DataFrame,
    randomness_stream: RandomnessStream,
    additional_key: str,
):
    """
    Makes vectorized choice for 2D array options.
    :param data: pd.Series which should be a subset of options.index
    :param options: pd.DataFrame where the index is the values of data and columns are available choices.
    :param randomness_stream: RandomnessStream object
    :param additional_key: key for randomness_stream
    :returns: pd.Series with new choices replacing the original values in data.
    """

    # Change columns to be integers for datawrangling later
    options.columns = list(range(len(options.columns)))
    # Get subset of options where we will choose new values
    data_idx = pd.Index(data.values)
    options = options.loc[data_idx]
    # Get number of options per name
    number_of_options = options.count(axis=1)

    # Find null values and calculate weights
    not_na = options.notna()
    row_weights = np.ones(len(number_of_options)) / number_of_options
    weights = not_na.mul(row_weights, axis=0)
    pmf = weights.div(weights.sum(axis=1), axis=0)
    cdf = np.cumsum(pmf, axis=1)
    # Get draw for each row
    probs = randomness_stream.get_draw(pd.Index(data.index), additional_key=additional_key)

    # Select indices of nickname to choose based on random draw
    choice_index = (probs.values[np.newaxis].T > cdf).sum(axis=1)
    options["choice_index"] = choice_index
    idx, cols = pd.factorize(options["choice_index"])
    # 2D array lookup to make an array for the series value
    new = pd.Series(
        options.reindex(cols, axis=1).to_numpy()[np.arange(len(options)), idx],
        index=data.index,
    )

    return new


def get_state_abbreviation(state: str) -> str:
    """
    Get the two letter abbreviation of a state in the US.

    :param state: A string representation of the state.
    :return: A string of length 2
    """
    state = state.upper()
    if state in metadata.US_STATE_ABBRV_MAP.values():
        return state
    try:
        return metadata.US_STATE_ABBRV_MAP[state]
    except KeyError:
        raise ValueError(f"Unexpected state input: '{state}'") from None


def cleanse_integer_columns(column: pd.Series) -> pd.Series:
    column = column.copy()
    column[column.notna()] = column[column.notna()].astype(str)
    float_mask = column.notna() & (column.astype(str).str.contains(".", regex=False))
    column.loc[float_mask] = column.loc[float_mask].astype(str).str.split(".").str[0]
    return column


##########################
# Data utility functions #
##########################


@cache
def load_ocr_errors_dict():
    ocr_errors = pd.read_csv(
        paths.OCR_ERRORS_DATA, skiprows=[0, 1], header=None, names=["ocr_true", "ocr_err"]
    )
    # Get OCR errors dict for noising
    ocr_error_dict = (
        ocr_errors.groupby("ocr_true")["ocr_err"].apply(lambda x: list(x)).to_dict()
    )

    return ocr_error_dict


@cache
def load_phonetic_errors_dict():
    phonetic_errors = pd.read_csv(
        paths.PHONETIC_ERRORS_DATA,
        skiprows=[0, 1],
        header=None,
        names=["where", "orig", "new", "pre", "post", "pattern", "start"],
    )
    phonetic_error_series = phonetic_errors.groupby("orig")["new"].apply(
        lambda x: list(x.str.replace("@", ""))
    )
    return phonetic_error_series.to_dict()


@cache
def load_qwerty_errors_data() -> pd.DataFrame:
    with open(paths.QWERTY_ERRORS) as f:
        qwerty_errors = yaml.safe_load(f)

    return pd.DataFrame.from_dict(qwerty_errors, orient="index")
