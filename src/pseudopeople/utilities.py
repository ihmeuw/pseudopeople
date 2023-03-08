from typing import Any, Union

import numpy as np
import pandas as pd

from vivarium.framework.lookup import LookupTable
from vivarium.framework.randomness import Array, RandomnessStream
from vivarium.framework.values import Pipeline

from pseudopeople.entities import Form
from pseudopeople.constants import paths


def get_randomness_stream(form: Form, seed: int) -> RandomnessStream:
    return RandomnessStream(form.value, lambda: pd.Timestamp("2020-04-01"), seed)


def vectorized_choice(
    options: Array,
    n_to_choose: int,
    randomness_stream: RandomnessStream = None,
    weights: Array = None,
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


def filter_by_rate(
    entity_to_filter: Union[pd.Index, pd.Series],
    randomness: RandomnessStream,
    rate_producer: Union[LookupTable, Pipeline],
    additional_key: Any = None,
) -> pd.Index:
    """
    Parameters
    ----------
    entity_to_filter: a series of every entity that might move. not necessarily unique.  Can be a list of ids or an
        index (household_ids, business_ids, or pandas index or simulants who may move).
    rate_producer: rate_producer for move rates
    randomness: RandomnessStream for component this is being run in
    additional_key: descriptive key to make sure randomness stream produces unique results
    Returns
    -------
    a pd.Index, subset from simulants, with those selected to be filtered.
    """
    entity_to_filter = entity_to_filter.drop_duplicates()
    if type(entity_to_filter) is pd.Series:
        idx = entity_to_filter.index
    else:
        idx = entity_to_filter

    filtered_sims = randomness.filter_for_rate(
        entity_to_filter, rate_producer(idx), additional_key
    )
    return filtered_sims
