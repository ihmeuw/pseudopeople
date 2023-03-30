from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

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
    noise_function: Callable[[pd.DataFrame, float, RandomnessStream, str], pd.DataFrame]

    def __call__(
        self,
        form_data: pd.DataFrame,
        configuration: float,
        randomness_stream: RandomnessStream,
    ) -> pd.DataFrame:
        return self.noise_function(form_data, configuration, randomness_stream)


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

    def __call__(
        self,
        column: pd.Series,
        configuration: ConfigTree,
        randomness_stream: RandomnessStream,
        additional_key: Any,
    ) -> pd.Series:
        column = column.copy()
        noise_level = configuration.row_noise_level
        to_noise_idx = get_index_to_noise(
            column, noise_level, randomness_stream, f"{self.name}_{additional_key}"
        )
        column.loc[to_noise_idx] = self.noise_function(
            column.loc[to_noise_idx], configuration, randomness_stream, additional_key
        )

        return column
