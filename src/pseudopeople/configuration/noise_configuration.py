from typing import Optional

from layered_config_tree import LayeredConfigTree

from pseudopeople.entity_types import ColumnNoiseType, NoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASET_SCHEMAS

COLUMN_NOISE_TYPES = [
    noise_type.name for noise_type in NOISE_TYPES if isinstance(noise_type, ColumnNoiseType)
]
ROW_NOISE_TYPES = [
    noise_type.name for noise_type in NOISE_TYPES if isinstance(noise_type, RowNoiseType)
]


class NoiseConfiguration:
    def __init__(self, config: LayeredConfigTree):
        self.config = config

    def get_parameter_value(
        self,
        dataset: str,
        noise_type: str,
        column_name: Optional[str] = None,
        parameter_name: Optional[str] = None,
    ):
        config = self.config
        try:
            dataset_config = config[dataset]
        except:
            raise ValueError(
                f"{dataset} not available in configuration."
                f"Available datasets are {list(config.keys())}"
            )

        def get_value_without_parameter_name(config: LayeredConfigTree, noise_type: str):
            if len(config[noise_type]) > 1:
                available_parameters = list(config[noise_type].keys())
                raise ValueError(
                    f"Your noising configuration has multiple parameters for the {dataset} dataset and noise type {noise_type}, so you must provide a parameter."
                    f"The available parameters are {available_parameters}"
                )
            else:
                parameter_name = list(config[noise_type].keys())[0]
                return config[noise_type][parameter_name]

        # row noise
        if noise_type in ROW_NOISE_TYPES:
            if column_name:
                raise ValueError(
                    f"You cannot provide both a row noise type ({noise_type}) and a column name ({column_name}) simultaneously."
                )

            if parameter_name:
                try:
                    return dataset_config["row_noise"][noise_type][parameter_name]
                except KeyError:
                    raise ValueError(
                        f"The parameter {parameter_name} was not found for {noise_type} in the configuration."
                        f"Available parameters are {list(dataset_config['row_noise'][noise_type].keys())}."
                    )
            else:
                parameter_value = get_value_without_parameter_name(
                    dataset_config["row_noise"], noise_type
                )
                return parameter_value

        # column noise
        elif noise_type in COLUMN_NOISE_TYPES:
            if not column_name:
                raise ValueError(
                    f"You must provide a column name when using a column noise type ({noise_type} in your case)."
                )
            try:
                column_config = dataset_config["column_noise"][column_name]
            except KeyError:
                raise ValueError(
                    f"The column name {column_name} was not found in your config."
                    f"Available columns are {list(dataset_config['column_noise'].keys())}."
                )
            if parameter_name:
                try:
                    return column_config[noise_type][parameter_name]
                except KeyError:
                    raise ValueError(
                        f"The parameter {parameter_name} was not found for {noise_type} in the configuration."
                        f"Available parameters are {list(dataset_config['column_noise'][column_name][noise_type].keys())}."
                    )
            else:
                parameter_value = get_value_without_parameter_name(column_config, noise_type)
                return parameter_value
        else:
            raise ValueError(
                f"Your noise type {noise_type} was not found in row noise types or column noise types. "
                f"Available row noise types are {ROW_NOISE_TYPES}. "
                f"Available column noise types are {COLUMN_NOISE_TYPES}."
            )
