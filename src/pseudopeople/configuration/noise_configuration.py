from typing import Any, Optional, Union

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
        self._config = config

    def to_dict(self) -> dict:
        config_dict: dict = self._config.to_dict()
        return config_dict

    def get_parameter_value(
        self,
        dataset: str,
        noise_type: str,
        column_name: Optional[str] = None,
        parameter_name: Optional[str] = None,
    ) -> Union[float, int]:
        config = self._config
        try:
            dataset_config = config[dataset]
        except:
            raise ValueError(
                f"{dataset} was not found in configuration. "
                f"Available datasets are {list(config.keys())}"
            )

        def _get_value_without_parameter_name(
            config: LayeredConfigTree, noise_type: str
        ) -> Union[int, float]:
            parameters: LayeredConfigTree = config[noise_type]
            if len(parameters) > 1:
                available_parameters = list(parameters.keys())
                raise ValueError(
                    f"Your noising configuration has multiple parameters for the {dataset} dataset and noise type {noise_type}, so you must provide a parameter. "
                    f"The available parameters are {available_parameters}"
                )
            else:
                parameter_name = list(parameters.keys())[0]
                value: Union[int, float] = config[noise_type][parameter_name]
                return value

        # row noise
        if noise_type in ROW_NOISE_TYPES:
            if column_name:
                raise ValueError(
                    f"You cannot provide both a row noise type ({noise_type}) and a column name ({column_name}) simultaneously."
                )

            if parameter_name:
                row_parameters: LayeredConfigTree = dataset_config["row_noise"][noise_type]
                try:
                    row_noise_value: Union[int, float] = row_parameters[parameter_name]
                    return row_noise_value
                except KeyError:
                    raise ValueError(
                        f"The parameter {parameter_name} was not found for {noise_type} in the configuration. "
                        f"Available parameters are {list(row_parameters.keys())}."
                    )
            else:
                row_noise_config: LayeredConfigTree = dataset_config["row_noise"]
                parameter_value = _get_value_without_parameter_name(
                    row_noise_config, noise_type
                )
                return parameter_value

        # column noise
        elif noise_type in COLUMN_NOISE_TYPES:
            if not column_name:
                raise ValueError(
                    f"You must provide a column name when using a column noise type ({noise_type} in your case)."
                )
            all_column_configs: LayeredConfigTree = dataset_config["column_noise"]
            try:
                column_config: LayeredConfigTree = all_column_configs[column_name]
            except KeyError:
                raise ValueError(
                    f"The column name {column_name} was not found in your config. "
                    f"Available columns are {list(all_column_configs.keys())}."
                )
            if parameter_name:
                column_parameters: LayeredConfigTree = column_config[noise_type]
                try:
                    column_noise_value: Union[int, float] = column_parameters[parameter_name]
                    return column_noise_value
                except KeyError:
                    raise ValueError(
                        f"The parameter {parameter_name} was not found for {noise_type} in the configuration. "
                        f"Available parameters are {list(column_parameters.keys())}."
                    )
            else:
                parameter_value = _get_value_without_parameter_name(column_config, noise_type)
                return parameter_value
        else:
            raise ValueError(
                f"Your noise type {noise_type} was not found in row noise types or column noise types. "
                f"Available row noise types are {ROW_NOISE_TYPES}. "
                f"Available column noise types are {COLUMN_NOISE_TYPES}."
            )

    def get_row_probability(self, dataset: str, noise_type: str) -> Union[int, float]:
        value: Union[int, float] = self.get_parameter_value(
            dataset, noise_type, parameter_name="row_probability"
        )
        return value

    def get_cell_probability(
        self, dataset: str, noise_type: str, column_name: str
    ) -> Union[int, float]:
        value: Union[int, float] = self.get_parameter_value(
            dataset, noise_type, column_name=column_name, parameter_name="cell_probability"
        )
        return value

    def get_token_probability(
        self, dataset: str, noise_type: str, column_name: str
    ) -> Union[int, float]:
        value: Union[int, float] = self.get_parameter_value(
            dataset, noise_type, column_name=column_name, parameter_name="token_probability"
        )
        return value
