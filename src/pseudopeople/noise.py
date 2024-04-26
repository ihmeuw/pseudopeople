"""
===================
   Noise Dataset
===================

A function that will noise data based on a user's configuration.

This function is the core of Pseudopeople, which takes in a pd.DataFrame or dataset
data and will add noise to both rows and columns based on the configured values
provided by the user.  First, the Dataset will be noised for missing data and have
rows in each columns changed to null values.  Then, the dataset data will be noised
by column and row for each type of additional noise type.
"""

from typing import Any

import pandas as pd
from layered_config_tree import LayeredConfigTree
from tqdm import tqdm

from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, Dataset
from pseudopeople.utilities import get_randomness_stream


def noise_dataset(
    dataset: Dataset,
    dataset_data: pd.DataFrame,
    configuration: LayeredConfigTree,
    seed: Any,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Adds noise to the input dataset data. Noise functions are executed in the order
    defined by :py:const: `.NOISE_TYPES`. Row noise functions are applied to the
    whole DataFrame. Column noise functions will be applied to each column that
    is pertinent to it.

    Noise levels are determined by the noise_config.
    :param dataset:
        Dataset needing to be noised
    :param dataset_data:
        Simulated population data which needs to be noised.
    :param configuration:
        Object to configure noise levels
    :param seed:
        Seed for controlling randomness
    :return:
        Noised dataset data
    """
    randomness = get_randomness_stream(dataset.name, seed, dataset_data.index)

    noise_configuration = configuration[dataset.name]

    # We only need to do this once, because noise does not introduce missingness,
    # except for the leave_blank kind which is special-cased below
    missingness = (dataset_data == "") | (dataset_data.isna())

    if progress_bar:
        noise_type_iterator = tqdm(
            NOISE_TYPES, desc="Applying noise", unit="type", leave=False
        )
    else:
        noise_type_iterator = NOISE_TYPES

    for noise_type in noise_type_iterator:
        if isinstance(noise_type, RowNoiseType):
            if (
                Keys.ROW_NOISE in noise_configuration
                and noise_type.name in noise_configuration.row_noise
            ):
                original_index = dataset_data.index
                # Apply row noise
                dataset_data = noise_type(
                    dataset.name,
                    dataset_data,
                    noise_configuration[Keys.ROW_NOISE][noise_type.name],
                    randomness,
                )
                missingness = missingness.loc[
                    dataset_data.index.intersection(original_index)
                ].copy()
                # Check for duplicated rows
                new_indices = dataset_data.index.difference(original_index)
                new_missingness = dataset_data.loc[new_indices].isna() | (
                    dataset_data.loc[new_indices] == ""
                )
                missingness = pd.concat([missingness, new_missingness])

        elif isinstance(noise_type, ColumnNoiseType):
            if Keys.COLUMN_NOISE in noise_configuration:
                columns_to_noise = [
                    col
                    for col in noise_configuration.column_noise
                    if col in dataset_data.columns
                    and noise_type.name in noise_configuration.column_noise[col]
                ]
                # Apply column noise to each column as appropriate
                for column in columns_to_noise:
                    required_cols = [column] + noise_type.additional_column_getter(column)
                    dataset_data[column], index_noised = noise_type(
                        dataset_data[required_cols],
                        noise_configuration.column_noise[column][noise_type.name],
                        randomness,
                        dataset.name,
                        column,
                        missingness=missingness[required_cols],
                    )
                    if noise_type == NOISE_TYPES.leave_blank:
                        # The only situation in which more missingness is introduced
                        missingness.loc[index_noised, column] = True
        else:
            raise TypeError(
                f"Invalid noise type. Allowed types are {RowNoiseType} and "
                f"{ColumnNoiseType}. Provided {type(noise_type)}."
            )

    return dataset_data
