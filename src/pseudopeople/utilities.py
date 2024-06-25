import sys
from dataclasses import dataclass
from functools import cache
import hashlib
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from pseudopeople.constants import metadata, paths
from pseudopeople.constants.noise_type_metadata import INT_TO_STRING_COLUMNS
from pseudopeople.dtypes import DtypeNames

if TYPE_CHECKING:
    from pseudopeople.dataset import Dataset
    from pseudopeople.schema_entities import DatasetSchema


def get_hash(key: str) -> int:
        max_allowable_numpy_seed = 4294967295  # 2**32 - 1
        return int(hashlib.sha1(key.encode("utf8")).hexdigest(), 16) % max_allowable_numpy_seed


def get_random_generator(dataset_name: str, seed: Any, index: pd.Index) -> np.random.default_rng:
    
    key = "_".join([dataset_name, str(seed)])
    return np.random.default_rng(get_hash(key))


def vectorized_choice(
    options: Union[list, pd.Series],
    n_to_choose: int,
    random_generator: np.random.default_rng,
    weights: Union[list, pd.Series] = None,
) -> np.ndarray:
    """
    Function that takes a list of options and uses Vivarium common random numbers framework to make a given number
    of random choice selections.

    :param options: List and series of possible values to choose
    :param n_to_choose: Number of choices to make, the length of the returned array of values
    :param random_generator: np.random.default_rng being used for common random numbers
    :param weights: List or series containing weights for each options

    returns: ndarray
    """
    # for each of n_to_choose, sample uniformly between 0 and 1
    index = pd.Index(np.arange(n_to_choose))
    probs = random_generator.random(size=len(index))

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
    dataset: "Dataset",
    noise_level: Union[float, pd.Series],
    required_columns: Optional[List[str]] = None,
) -> pd.Index:
    """
    Function that takes a series and returns a pd.Index that chosen by Vivarium Common Random Number to be noised.
    """

    index_eligible_for_noise = dataset.get_non_empty_index(required_columns)
    
    if isinstance(noise_level, float):
        number_to_noise = dataset.randomness.binomial(len(index_eligible_for_noise), p=noise_level)
        to_noise_idx = pd.Index(
            dataset.randomness.choice(index_eligible_for_noise, size=number_to_noise, replace=False)
        )
    else:
        # This is a copy paste of filter for probability
        draws = dataset.randomness.random(size=len(index_eligible_for_noise))
        chosen = np.array(draws < noise_level)
        to_noise_idx = index_eligible_for_noise[chosen]

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
    random_generator: np.random.default_rng,
    additional_key: str,
):
    """
    Makes vectorized choice for 2D array options.
    :param data: pd.Series which should be a subset of options.index
    :param options: pd.DataFrame where the index is the values of data and columns are available choices.
    :param random_generator: np.random.default_rng instance
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
    probs = random_generator.random(pd.Index(data.index), additional_key=additional_key)

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


def to_string_preserve_nans(s: pd.Series) -> pd.Series:
    # NOTE: In newer versions of pandas, astype(str) will use the *pandas*
    # string type, which we haven't adopted yet.
    result = s.astype(str).astype(DtypeNames.OBJECT)
    result[s.isna()] = np.nan
    return result


def to_string_as_integer(column: pd.Series) -> pd.Series:
    column = to_string_preserve_nans(column)
    float_mask = column.notna() & (column.str.contains(".", regex=False))
    column.loc[float_mask] = column.loc[float_mask].astype(str).str.split(".").str[0]
    return column


def to_string(column: pd.Series) -> pd.Series:
    if column.name in INT_TO_STRING_COLUMNS:
        return to_string_as_integer(column)
    else:
        return to_string_preserve_nans(column)


def ensure_dtype(data: pd.Series, dtype: np.dtype):
    if dtype.name == DtypeNames.OBJECT:
        return to_string(data)
    else:
        return data.astype(dtype)


def count_number_of_tokens_per_string(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """
    Calculates the number of tokens in each string of a series.
    s1 is a pd.Series of tokens and we want to count how many tokens exist in each
    string of s2. That is if s1 were a series of length 2 and s2 were a series of
    length 1, we would return a series that is length 1 with the total count of
    s1[0] in s2 and s1[1] in s2.
    """

    s2 = s2.astype(str)
    strings = s2.unique()
    tokens_per_string = pd.Series(
        (sum(count_occurrences(s, str(token)) for token in s1) for s in strings),
        index=strings,
    )

    number_of_tokens = s2.map(tokens_per_string)
    number_of_tokens.index = s2
    return number_of_tokens


# https://stackoverflow.com/a/2970542/
def count_occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def coerce_dtypes(
    data: pd.DataFrame,
    dataset_schema: "DatasetSchema",
) -> pd.DataFrame:
    for col in dataset_schema.columns:
        if col.dtype_name != data[col.name].dtype.name:
            if col.dtype_name == DtypeNames.OBJECT:
                data[col.name] = to_string(data[col.name])
            else:
                data[col.name] = data[col.name].astype(col.dtype_name)

    return data


####################
# Engine utilities #
####################


@dataclass
class Engine:
    name: str
    dataframe_class_getter: Callable

    @property
    def dataframe_class(self):
        return self.dataframe_class_getter()


PANDAS_ENGINE = Engine("pandas", lambda: pd.DataFrame)


def get_dask_dataframe():
    import dask.dataframe as dd

    return dd.DataFrame


DASK_ENGINE = Engine("dask", get_dask_dataframe)


def get_engine_from_string(engine: str) -> Engine:
    if engine == "pandas":
        return PANDAS_ENGINE
    elif engine == "dask":
        return DASK_ENGINE
    else:
        raise ValueError(f"Unknown engine {engine}")


try:
    # Optional dependency
    import dask.dataframe as dd

    DataFrame = Union[dd.DataFrame, pd.DataFrame]
except ImportError:
    DataFrame = pd.DataFrame


##########################
# Data utility functions #
##########################


@cache
def load_ocr_errors():
    ocr_errors = pd.read_csv(
        paths.OCR_ERRORS_DATA, skiprows=[0, 1], header=None, names=["ocr_true", "ocr_err"]
    )
    # Get OCR errors dict for noising
    ocr_error_dict = (
        ocr_errors.groupby("ocr_true")["ocr_err"].apply(lambda x: list(x)).to_dict()
    )
    ocr_errors = pd.DataFrame.from_dict(ocr_error_dict, orient="index")

    return ocr_errors


@cache
def load_phonetic_errors():
    phonetic_errors = pd.read_csv(
        paths.PHONETIC_ERRORS_DATA,
        skiprows=[0, 1],
        header=None,
        names=["where", "orig", "new", "pre", "post", "pattern", "start"],
    )
    phonetic_error_dict = (
        phonetic_errors.groupby("orig")["new"]
        .apply(lambda x: list(x.str.replace("@", "")))
        .to_dict()
    )
    phonetic_errors = pd.DataFrame.from_dict(phonetic_error_dict, orient="index")

    return phonetic_errors


@cache
def load_qwerty_errors_data() -> pd.DataFrame:
    with open(paths.QWERTY_ERRORS) as f:
        qwerty_errors = yaml.safe_load(f)

    return pd.DataFrame.from_dict(qwerty_errors, orient="index")
