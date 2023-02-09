from dataclasses import dataclass
from typing import Callable, List

import pandas as pd

from pseudo_people.configuration import (
    ColumnNoiseConfigurationNode,
    RowNoiseConfigurationNode,
)


@dataclass
class RowNoiseType:
    """
    Defines a type of noise that can be applied to a row.

    The name is the name of the particular noise function (e.g. "omission" or
    "duplication").

    The noise function takes as input a DataFrame and the configuration object
    for this RowNoise operation. It applies the noising operation to the entire
    DataFrame and returns the modified DataFrame.
    """

    name: str
    noise_function: Callable[[pd.DataFrame, RowNoiseConfigurationNode], pd.DataFrame]

    def __call__(
        self, form_data: pd.DataFrame, configuration: RowNoiseConfigurationNode
    ) -> pd.DataFrame:
        return self.noise_function(form_data, configuration)


@dataclass
class ColumnNoiseType:
    """
    Defines a type of noise that can be applied to a column.

    The name is the name of the particular noise function (e.g. "nickname" or
    "phonetic").

    The noise function takes as input a Series and the configuration object
    for this ColumnNoise operation. It applies the noising operation to the
    Series and returns the modified Series.
    """

    name: str
    noise_function: Callable[[pd.Series, ColumnNoiseConfigurationNode], pd.Series]

    def __call__(
        self, column: pd.Series, configuration: ColumnNoiseConfigurationNode
    ) -> pd.Series:
        return self.noise_function(column, configuration)


@dataclass
class ColumnMetadata:
    """
    Defines a column and its noise functions.

    The name is the name of the particular column (e.g. "first_name" or
    "state").

    The noise types is a list of the types of noise that should be applied to
    this column in the order of application.
    """

    name: str
    noise_types: List[ColumnNoiseType]
