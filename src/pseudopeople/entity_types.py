from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger
from vivarium import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.utilities import get_index_to_noise

if TYPE_CHECKING:
    from pseudopeople.dataset import DatasetData


@dataclass
class NoiseType(ABC):
    name: str
    noise_function: Callable[["DatasetData", ConfigTree, pd.Index], None]
    probability: Optional[float] = 0.0
    additional_parameters: Dict[str, Any] = None

    @abstractmethod
    def probability_key(self) -> str:
        pass


@dataclass
class RowNoiseType(NoiseType):
    """
    Defines a type of noise that can be applied to a row.

    The name is the name of the particular noise type (e.g. "omit_row" or
    "duplicate_row").

    The noise function takes as input a DataFrame, the configuration value
    for this RowNoise operation, and a RandomnessStream for controlling
    randomness. It applies the noising operation to the entire DataFrame and
    returns the modified DataFrame.
    """

    get_noise_level: Callable[["DatasetData", ConfigTree], float] = lambda _, config: config[
        Keys.ROW_PROBABILITY
    ]

    @property
    def probability_key(self) -> str:
        return Keys.ROW_PROBABILITY

    def __call__(self, dataset_data: "DatasetData", configuration: ConfigTree) -> None:
        noise_level = self.get_noise_level(dataset_data, configuration)
        to_noise_idx = get_index_to_noise(dataset_data, noise_level, self.name)
        self.noise_function(dataset_data, configuration, to_noise_idx)


@dataclass
class ColumnNoiseType(NoiseType):
    """
    Defines a type of noise that can be applied to a column.

    The name is the name of the particular noise type (e.g. use_nickname" or
    "make_phonetic_errors").

    The noise function takes as input a DataFrame, the ConfigTree object for this
    ColumnNoise operation, a RandomnessStream for controlling randomness, and
    a column name, which is the column that will be noised and who's name will be used
    as the additional key for the RandomnessStream.
    Optionally, it can take a pre-existing DataFrame indicating where there is missingness
    in the data (same index and columns as the main DataFrame, all boolean type) --
    if this is not passed, it calculates it, which can be expensive for large data.
    It applies the noising operation to the Series and returns both the modified Series
    and an Index of which items in the Series were selected for noise.
    """

    noise_function: Callable[["DatasetData", ConfigTree, pd.Index, Optional[str]], None]
    probability: Optional[float] = 0.01
    noise_level_scaling_function: Callable[[pd.DataFrame, str], float] = lambda x, y: 1.0
    additional_column_getter: Callable[[str], List[str]] = lambda column_name: []

    @property
    def probability_key(self) -> str:
        return Keys.CELL_PROBABILITY

    def __call__(
        self,
        dataset_data: "DatasetData",
        configuration: ConfigTree,
        column_name: str,
    ) -> None:
        if dataset_data.is_empty(column_name):
            return

        noise_level = configuration[
            Keys.CELL_PROBABILITY
        ] * self.noise_level_scaling_function(dataset_data.data, column_name)

        # Certain columns have their noise level scaled so we must check to make
        # sure the noise level is within the allowed range between 0 and 1 for
        # probabilities
        noise_level = min(noise_level, 1.0)
        to_noise_idx = get_index_to_noise(
            dataset_data,
            noise_level,
            f"{self.name}_{column_name}",
            [column_name] + self.additional_column_getter(column_name),
        )
        if to_noise_idx.empty:
            logger.debug(
                f"No cells chosen to noise for noise function {self.name} on "
                f"column {column_name}. "
                "This is likely due to a combination "
                "of the configuration noise levels and the simulated population "
                "data."
            )
            return

        original_dtype_name = dataset_data.data[column_name].dtype.name
        self.noise_function(
            dataset_data,
            configuration,
            to_noise_idx,
            column_name,
        )

        # todo investigate this and also move the logic inside DatasetData
        # Coerce noised column dtype back to original column's if it has changed
        if dataset_data.data[column_name].dtype.name != original_dtype_name:
            dataset_data = dataset_data.data[column_name].astype(
                dataset_data.data[column_name][column_name].dtype
            )
