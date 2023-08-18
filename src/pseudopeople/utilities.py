from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
import pandas as pd
from loguru import logger
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap

from pseudopeople.constants import metadata, paths

if TYPE_CHECKING:
    from pseudopeople.utilities import DATAFRAME, ENGINE, INDEX


def get_randomness_stream(dataset_name: str, seed: int, index: pd.Index) -> RandomnessStream:
    map_size = max(1_000_000, max(index) * 2)
    return RandomnessStream(
        key=dataset_name,
        clock=lambda: pd.Timestamp("2020-04-01"),
        seed=seed,
        index_map=IndexMap(size=map_size),
    )


def vectorized_choice(
    options: Union[list, SERIES],
    n_to_choose: Union[int, DATAFRAME],
    randomness_stream: RandomnessStream,
    weights: Union[list, SERIES] = None,
    additional_key: Any = None,
    engine: ENGINE = "pandas",
):
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
    numpy_like, pandas_like = engine_implementations(engine)

    if weights is None:
        n = len(options)
        weights = numpy_like.ones(n) / n
    if isinstance(weights, list):
        weights = numpy_like.array(weights)

    # for each of n_to_choose, sample uniformly between 0 and 1
    index = pandas_like.Index(numpy_like.arange(n_to_choose))
    probs = randomness_stream.get_draw(index, additional_key=additional_key)

    # build cdf based on weights
    pmf = weights / weights.sum()
    cdf = numpy_like.cumsum(pmf)

    # for each p_i in probs, count how many elements of cdf for which p_i >= cdf_i
    chosen_indices = numpy_like.searchsorted(cdf, probs, side="right")

    if engine == "pandas":
        # We generally assume numpy will be faster
        return numpy_like.take(options, chosen_indices)
    else:
        # Because we aren't distributing numpy, we want to get back to pandas-like
        # sooner
        return pandas_like.Series(chosen_indices).map({i: o for i, o in enumerate(options)})


def get_index_to_noise(
    data: DATAFRAME,
    noise_level: Union[SERIES, float],
    randomness_stream: RandomnessStream,
    additional_key: Any,
    is_column_noise: bool = False,
) -> INDEX:
    """
    Function that takes a series and returns an index that chosen by Vivarium Common Random Number to be noised.
    """

    # Get rows to noise
    if is_column_noise:
        missing_idx = data.index[(data.isna().any(axis=1)) | (data.isin([""]).any(axis=1))]
        eligible_for_noise_idx = data.index.difference(missing_idx)
    else:
        # Any index can be noised for row noise
        eligible_for_noise_idx = data.index

    # TODO: Currently we must go to pandas before we can interact with randomness_stream
    if not isinstance(noise_level, float) and engine_from_object(noise_level) == "modin":
        noise_level = noise_level._to_pandas()

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
    data: SERIES,
    options: DATAFRAME,
    randomness_stream: RandomnessStream,
    additional_key: str,
) -> SERIES:
    """
    Makes vectorized choice for 2D array options.
    :param data: Series which should be a subset of options.index
    :param options: DataFrame where the index is the values of data and columns are available choices.
    :param randomness_stream: RandomnessStream object
    :param additional_key: key for randomness_stream
    :returns: Series with new choices replacing the original values in data.
    """
    engine, numpy_like, pandas_like = engine_implementations_from_object(data)

    # Change columns to be integers for datawrangling later
    options.columns = list(range(len(options.columns)))
    # Get subset of options where we will choose new values
    data_idx = pandas_like.Index(data)
    options = options.loc[data_idx]
    # Get number of options per name
    number_of_options = options.count(axis=1)

    # Find null values and calculate weights
    not_na = options.notna()
    row_weights = numpy_like.ones(len(number_of_options)) / number_of_options
    weights = not_na.mul(row_weights, axis=0)
    pmf = weights.div(weights.sum(axis=1), axis=0)
    cdf = numpy_like.cumsum(pmf, axis=1)
    # Get draw for each row
    # TODO: This will currently return a pandas Series, no matter what engine we
    # are using.
    probs = randomness_stream.get_draw(data.index, additional_key=additional_key)

    # Select indices of nickname to choose based on random draw
    choice_index = (probs.values[np.newaxis].T > cdf).sum(axis=1)
    options["choice_index"] = choice_index
    idx, cols = pandas_like.factorize(options["choice_index"])
    # 2D array lookup to make an array for the series value
    new = pandas_like.Series(
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


ENGINE = Literal["pandas", "modin"]

if TYPE_CHECKING:
    import modin.pandas as mpd

    DATAFRAME = Union[mpd.DataFrame, pd.DataFrame]
    SERIES = Union[mpd.Series, pd.Series]
    INDEX = Union[mpd.Index, pd.Index]


def engine_from_object(obj: Union[DATAFRAME, SERIES, INDEX]) -> ENGINE:
    if (
        isinstance(obj, pd.DataFrame)
        or isinstance(obj, pd.Series)
        or isinstance(obj, pd.Index)
    ):
        return "pandas"

    try:
        import modin.pandas as mpd

        if (
            isinstance(obj, mpd.DataFrame)
            or isinstance(obj, mpd.Series)
            or isinstance(obj, mpd.Index)
        ):
            return "modin"
    except ImportError:
        pass

    try:
        import dask.dataframe as dd

        if (
            isinstance(obj, dd.DataFrame)
            or isinstance(obj, dd.Series)
            or isinstance(obj, dd.Index)
        ):
            return "dask"
    except ImportError:
        pass

    raise ValueError("Unknown object type")


def engine_implementations(engine: ENGINE):
    # modin.numpy is not good enough yet
    numpy_like = np

    if engine == "pandas":
        pandas_like = pd
    else:
        try:
            import modin.pandas as mpd
        except ImportError:
            raise ImportError(
                "Optional dependency modin is not installed, and is required for this operation."
            ) from None

        pandas_like = mpd

    return numpy_like, pandas_like


def engine_implementations_from_object(obj: Union[DATAFRAME, SERIES, INDEX]):
    engine = engine_from_object(obj)

    return (engine,) + engine_implementations(engine)


##########################
# Data utility functions #
##########################


def load_ocr_errors_dict():
    ocr_errors = pd.read_csv(
        paths.OCR_ERRORS_DATA, skiprows=[0, 1], header=None, names=["ocr_true", "ocr_err"]
    )
    # Get OCR errors dict for noising
    ocr_error_dict = (
        ocr_errors.groupby("ocr_true")["ocr_err"].apply(lambda x: list(x)).to_dict()
    )

    return ocr_error_dict


def load_phonetic_errors_dict():
    phonetic_errors = pd.read_csv(
        paths.PHONETIC_ERRORS_DATA,
        skiprows=[0, 1],
        header=None,
        names=["where", "orig", "new", "pre", "post", "pattern", "start"],
    )
    phonetic_error_dict = phonetic_errors.groupby("orig")["new"].apply(
        lambda x: list(x.str.replace("@", ""))
    )
    return phonetic_error_dict
