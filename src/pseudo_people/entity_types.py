from dataclasses import dataclass
from typing import Callable

import pandas as pd
from vivarium import ConfigTree


@dataclass
class RowNoiseType:
    """
    Defines a type of noise that can be applied to a row.

    The name is the name of the particular noise function (e.g. "omission" or
    "duplication").

    The noise function takes as input a DataFrame and the configuration value
    for this RowNoise operation. It applies the noising operation to the entire
    DataFrame and returns the modified DataFrame.
    """

    name: str
    noise_function: Callable[[pd.DataFrame, float], pd.DataFrame]

    def __call__(self, form_data: pd.DataFrame, configuration: float) -> pd.DataFrame:
        return self.noise_function(form_data, configuration)


@dataclass
class ColumnNoiseType:
    """
    Defines a type of noise that can be applied to a column.

    The name is the name of the particular noise function (e.g. "nickname" or
    "phonetic").

    The noise function takes as input a Series and the ConfigTree object for
    this ColumnNoise operation. It applies the noising operation to the Series
    and returns the modified Series.
    """

    name: str
    noise_function: Callable[[pd.Series, ConfigTree], pd.Series]

    def __call__(self, column: pd.Series, configuration: ConfigTree) -> pd.Series:
        return self.noise_function(column, configuration)
