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

    def get_value(
        self,
        dataset: str,
        noise_type: str,
        parameter_name: str,
        column_name: Optional[str] = None,
    ) -> Union[float, int]:
        config = self._config
        try:
            dataset_config = config[dataset]
        except:
            raise ValueError(
                f"{dataset} was not found in configuration. "
                f"Available datasets are {list(config.keys())}"
            )

        # row noise
        if noise_type in ROW_NOISE_TYPES:
            if column_name:
                raise ValueError(
                    f"You cannot provide both a row noise type ({noise_type}) and a column name ({column_name}) simultaneously."
                )
            config = dataset_config["row_noise"]
        # column noise
        elif noise_type in COLUMN_NOISE_TYPES:
            if not column_name:
                raise ValueError(
                    f"You must provide a column name when using a column noise type ({noise_type} in your case)."
                )
            all_column_configs: LayeredConfigTree = dataset_config["column_noise"]
            if column_name not in all_column_configs:
                raise ValueError(
                    f"The column name {column_name} was not found in your config. "
                    f"Available columns are {list(all_column_configs.keys())}."
                )
            config = all_column_configs[column_name]
        # unknown noise type
        else:
            raise ValueError(
                f"Your noise type {noise_type} was not found in row noise types or column noise types. "
                f"Available row noise types are {ROW_NOISE_TYPES}. "
                f"Available column noise types are {COLUMN_NOISE_TYPES}."
            )
        # get value
        parameter_tree: LayeredConfigTree = config[noise_type]
        if parameter_name not in parameter_tree:
            raise ValueError(
                f"The parameter {parameter_name} was not found for {noise_type} in the configuration. "
                f"Available parameters are {list(parameter_tree.keys())}."
            )
        noise_value: Union[int, float] = parameter_tree[parameter_name]
        return noise_value

    def get_row_probability(self, dataset: str, noise_type: str) -> Union[int, float]:
        value: Union[int, float] = self.get_value(
            dataset, noise_type, parameter_name="row_probability"
        )
        return value

    def get_cell_probability(
        self, dataset: str, noise_type: str, column_name: str
    ) -> Union[int, float]:
        value: Union[int, float] = self.get_value(
            dataset, noise_type, parameter_name="cell_probability", column_name=column_name
        )
        return value

    def get_token_probability(
        self, dataset: str, noise_type: str, column_name: str
    ) -> Union[int, float]:
        value: Union[int, float] = self.get_value(
            dataset, noise_type, parameter_name="token_probability", column_name=column_name
        )
        return value

    def has_row_noise_type(self, dataset_name: str, noise_type: str) -> bool:
        dataset_config = self.to_dict()[dataset_name]
        has_row_noise_type = (
            "row_noise" in dataset_config and noise_type in dataset_config["row_noise"]
        )
        return has_row_noise_type
