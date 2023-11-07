from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    The name is the name of the particular noise type (e.g. "omit_row" or
    "duplicate_row").

    The noise function takes as input a DataFrame, the configuration value
    for this RowNoise operation, and a RandomnessStream for controlling
    randomness. It applies the noising operation to the entire DataFrame and
    returns the modified DataFrame.
    """

    name: str
    noise_function: Callable[[str, pd.DataFrame, ConfigTree, RandomnessStream], pd.DataFrame]
    row_probability: float = 0.0

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

    name: str
    noise_function: Callable[[pd.Series, ConfigTree, RandomnessStream, Any], pd.Series]
    cell_probability: Optional[float] = 0.01
    noise_level_scaling_function: Callable[[pd.DataFrame, str], float] = lambda x, y: 1.0
    additional_parameters: Dict[str, Any] = None
    additional_column_getter: Callable[[str], List[str]] = lambda column_name: []

    def __call__(
        self,
        data: pd.DataFrame,
        configuration: ConfigTree,
        randomness_stream: RandomnessStream,
        dataset_name: str,
        column_name: str,
        missingness: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Index]:
        if data[column_name].empty:
            return data[column_name], pd.Index([])

        noise_level = configuration[
            Keys.CELL_PROBABILITY
        ] * self.noise_level_scaling_function(data, column_name)
        # Certain columns have their noise level scaled so we must check to make sure the noise level is within the
        # allowed range between 0 and 1 for probabilities
        noise_level = min(noise_level, 1.0)
        to_noise_idx = get_index_to_noise(
            data,
            noise_level,
            randomness_stream,
            f"{self.name}_{column_name}",
            is_column_noise=True,
            missingness=missingness,
        )
        if to_noise_idx.empty:
            logger.debug(
                f"No cells chosen to noise for noise function {self.name} on column {column_name}. "
                "This is likely due to a combination of the configuration noise levels and the simulated population data."
            )
            return data[column_name], to_noise_idx
        noised_data = self.noise_function(
            data.loc[to_noise_idx],
            configuration,
            randomness_stream,
            dataset_name,
            column_name,
        )

        # Coerce noised column dtype back to original column's if it has changed
        if noised_data.dtype.name != data[column_name].dtype.name:
            noised_data = noised_data.astype(data[column_name].dtype)

        result = data[column_name].copy()
        result.loc[to_noise_idx] = noised_data

        return result, to_noise_idx
