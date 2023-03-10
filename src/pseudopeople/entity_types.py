from dataclasses import dataclass
from typing import Callable

import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream


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
    noise_function: Callable[[pd.DataFrame, float, RandomnessStream], pd.DataFrame]

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
    ColumnNoise operation, and a RandomnessStream for controlling randomness. It
    applies the noising operation to the Series and returns the modified Series.
    """

    name: str
    noise_function: Callable[[pd.Series, ConfigTree, RandomnessStream], pd.Series]

    def __call__(
        self,
        column: pd.Series,
        configuration: ConfigTree,
        randomness_stream: RandomnessStream,
    ) -> pd.Series:
        return self.noise_function(column, configuration, randomness_stream)
