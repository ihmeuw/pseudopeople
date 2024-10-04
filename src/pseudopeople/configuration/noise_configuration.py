from __future__ import annotations

from typing import Any, Optional, Union

from layered_config_tree import LayeredConfigTree
from layered_config_tree.types import InputData

from pseudopeople.configuration import Keys
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
    ) -> Union[float, int, list, dict]:
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
        noise_value: Union[int, float, LayeredConfigTree] = parameter_tree[parameter_name]
        # TODO: [MIC-5238] deal with properly when updating column noising, possibly with custom getter
        converted_noise_value: Union[int, float, dict] = (
            noise_value.to_dict()
            if isinstance(noise_value, LayeredConfigTree)
            else noise_value
        )
        return converted_noise_value

    def set_value(
        self,
        dataset: str,
        noise_type: str,
        parameter_name: str,
        new_value: Union[float, int, list, dict],
        column_name: Optional[str] = None,
    ) -> None:
        if parameter_name == Keys.POSSIBLE_AGE_DIFFERENCES:
            if isinstance(new_value, dict) or isinstance(new_value, list):
                new_value = self._format_misreport_age_perturbations(dataset, new_value)
        if column_name is not None:
            updated_tree = LayeredConfigTree(
                {
                    dataset: {
                        "column_noise": {
                            column_name: {noise_type: {parameter_name: new_value}}
                        }
                    }
                }
            )
        else:
            updated_tree = LayeredConfigTree(
                {dataset: {"row_noise": {noise_type: {parameter_name: new_value}}}}
            )

        self._config.update(updated_tree)

    def update(self, data: InputData) -> None:
        self._config.update(data)

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

    def has_noise_type(
        self, dataset_name: str, noise_type: str, column: str | None = None
    ) -> bool:
        dataset_config = self.to_dict()[dataset_name]
        if column is not None:
            has_noise_type = noise_type in dataset_config.get("column_noise", {}).get(
                column, {}
            )
        else:
            has_noise_type = noise_type in dataset_config.get("row_noise", {})
        return has_noise_type

    def _format_misreport_age_perturbations(
        self, dataset: str, perturbations: Union[list[int], dict[int, float]]
    ) -> dict[int, float]:
        # Format any age perturbation lists as a dictionary with uniform probabilities
        formatted = {}
        default_perturbations: dict[int, float] = self.get_value(
            dataset, NOISE_TYPES.misreport_age.name, Keys.POSSIBLE_AGE_DIFFERENCES, "age"
        )
        # Replace default configuration with 0 probabilities
        for perturbation in default_perturbations:
            formatted[perturbation] = 0.0
        if isinstance(perturbations, list):
            # Add user perturbations with uniform probabilities
            uniform_prob = 1 / len(perturbations)
            for perturbation in perturbations:
                formatted[perturbation] = uniform_prob
        else:
            for perturbation, prob in perturbations.items():
                formatted[perturbation] = prob

        return formatted
