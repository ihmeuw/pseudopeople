from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd
from loguru import logger
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.configuration import Keys
from pseudopeople.utilities import get_index_to_noise


@dataclass
class RowNoiseType:
    """
    Defines a type of noise that can be applied to a row.

    The name is the name of the particular noise function (e.g. "omission" or
    "duplication").

    The noise function takes as input a DataFrame, the configuration value
    for this RowNoise operation, and a RandomnessStream for controlling
    randomness. It applies the noising operation to the entire DataFrame and
    returns the modified DataFrame.
    """

    name: str
    noise_function: Callable[[str, pd.DataFrame, ConfigTree, RandomnessStream], pd.DataFrame]
    probability: float = 0.0

    def __call__(
        self,
        dataset_name: str,
        dataset_data: pd.DataFrame,
        configuration: ConfigTree,
        randomness_stream: RandomnessStream,
    ) -> pd.DataFrame:
        return self.noise_function(
            dataset_name, dataset_data, configuration, randomness_stream
        )


@dataclass
class ColumnNoiseType:
    """
    Defines a type of noise that can be applied to a column.

    The name is the name of the particular noise function (e.g. "nickname" or
    "phonetic").

    The noise function takes as input a Series, the ConfigTree object for this
    ColumnNoise operation, a RandomnessStream for controlling randomness, and
    an additional key for the RandomnessStream. It applies the noising operation
    to the Series and returns the modified Series.
    """

    name: str
    noise_function: Callable[[pd.Series, ConfigTree, RandomnessStream, Any], pd.Series]
    probability: Optional[float] = 0.01
    noise_level_scaling_function: Callable[[str], float] = lambda x: 1.0
    additional_parameters: Dict[str, Any] = None

    def __call__(
        self,
        column: pd.Series,
        configuration: ConfigTree,
        randomness_stream: RandomnessStream,
        additional_key: Any,
    ) -> pd.Series:
        column = column.copy()
        probability_key = (
            Keys.CELL_PROBABILITY
            if Keys.CELL_PROBABILITY in configuration.keys()
            else Keys.PROBABILITY
        )
        noise_level = configuration[probability_key] * self.noise_level_scaling_function(
            column.name
        )
        to_noise_idx = get_index_to_noise(
            column, noise_level, randomness_stream, f"{self.name}_{additional_key}"
        )
        if to_noise_idx.empty:
            logger.debug(
                f"No cells chosen to noise for noise function {self.name} on column {column.name}. "
                "This is likely due to a combination of the configuration noise levels and the input data."
            )
            return column
        noised_data = self.noise_function(
            column.loc[to_noise_idx], configuration, randomness_stream, additional_key
        )

        # Coerce noised column dtype back to original column's if it has changed
        if noised_data.dtype.name != column.dtype.name:
            noised_data = noised_data.astype(column.dtype)

        column.loc[to_noise_idx] = noised_data

        return column
