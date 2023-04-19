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
import pandas as pd
from tqdm import tqdm
from vivarium import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import Dataset
from pseudopeople.utilities import get_randomness_stream


def noise_dataset(
    dataset: Dataset,
    dataset_data: pd.DataFrame,
    configuration: ConfigTree,
    seed: int,
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
        Clean data input which needs to be noised.
    :param configuration:
        Object to configure noise levels
    :param seed:
        Seed for controlling randomness
    :return:
        Noised dataset data
    """
    randomness = get_randomness_stream(dataset.name, seed)

    noise_configuration = configuration[dataset.name]
    for noise_type in tqdm(NOISE_TYPES, desc="Applying noise", unit="type"):
        if isinstance(noise_type, RowNoiseType):
            if (
                Keys.ROW_NOISE in noise_configuration
                and noise_type.name in noise_configuration.row_noise
            ):
                # Apply row noise
                dataset_data = noise_type(
                    dataset.name,
                    dataset_data,
                    noise_configuration[Keys.ROW_NOISE][noise_type.name],
                    randomness,
                )

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
                    dataset_data[column] = noise_type(
                        dataset_data[column],
                        noise_configuration.column_noise[column][noise_type.name],
                        randomness,
                        column,
                    )
        else:
            raise TypeError(
                f"Invalid noise type. Allowed types are {RowNoiseType} and "
                f"{ColumnNoiseType}. Provided {type(noise_type)}."
            )

    return dataset_data
