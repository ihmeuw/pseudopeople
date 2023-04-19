from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from vivarium.config_tree import ConfigTree, ConfigurationKeyError

from pseudopeople.configuration import Keys


@dataclass
class ConfigurationError(BaseException):
    """Base class for configuration errors"""

    message: str


def validate_user_configuration(user_config: Dict, default_config: ConfigTree) -> None:
    """
    Validates the user-provided configuration. Confirms that all user-provided
    keys exist in the default configuration. Confirms that all user-provided
    values are valid for their respective noise functions.
    """
    for dataset, dataset_config in user_config.items():
        default_dataset_config = _get_default_config_node(default_config, dataset, "dataset")
        for key in dataset_config:
            _get_default_config_node(
                default_dataset_config, key, "configuration key", dataset
            )

        default_row_noise_config = default_dataset_config[Keys.ROW_NOISE]
        default_column_noise_config = default_dataset_config[Keys.COLUMN_NOISE]

        for noise_type, noise_type_config in dataset_config.get(Keys.ROW_NOISE, {}).items():
            default_noise_type_config = _get_default_config_node(
                default_row_noise_config, noise_type, "noise type", dataset
            )
            _validate_noise_type_config(
                noise_type_config, default_noise_type_config, dataset, noise_type
            )

        for column, column_config in dataset_config.get(Keys.COLUMN_NOISE, {}).items():
            default_column_config = _get_default_config_node(
                default_column_noise_config, column, "column", dataset
            )
            for noise_type, noise_type_config in column_config.items():
                default_noise_type_config = _get_default_config_node(
                    default_column_config, noise_type, "noise type", dataset, column
                )
                _validate_noise_type_config(
                    noise_type_config, default_noise_type_config, dataset, noise_type, column
                )


def _validate_noise_type_config(
    noise_type_config: Union[Dict, List],
    default_noise_type_config: ConfigTree,
    dataset: str,
    noise_type: str,
    column: str = None,
) -> None:
    """
    Validates that all parameters are allowed for this noise function.
    Additionally, validates that the configuration values are permissible.
    """
    for parameter, parameter_config in noise_type_config.items():
        parameter_config_validator = {
            Keys.POSSIBLE_AGE_DIFFERENCES: _validate_possible_age_differences,
            Keys.ZIPCODE_DIGIT_PROBABILITIES: _validate_zipcode_digit_probabilities,
        }.get(parameter, _validate_probability)

        _ = _get_default_config_node(
            default_noise_type_config, parameter, "parameter", dataset, column, noise_type
        )
        base_error_message = (
            f"Invalid '{parameter}' provided for dataset '{dataset}' for "
            f"column '{column}' and noise type '{noise_type}'. "
        )
        parameter_config_validator(parameter_config, parameter, base_error_message)


def _get_default_config_node(
    default_config: ConfigTree,
    key: str,
    key_type: str,
    dataset: str = None,
    column: str = None,
    noise_type: str = None,
) -> ConfigTree:
    """
    Validate that the node the user is trying to add exists in the default
    configuration.
    """
    try:
        return default_config[key]
    except ConfigurationKeyError:
        dataset_context = "" if dataset is None else f" for dataset '{dataset}'"
        column_context = "" if column is None else f" for column '{column}'"
        noise_type_context = "" if noise_type is None else f" and noise type '{noise_type}'"
        context = dataset_context + column_context + noise_type_context

        error_message = f"Invalid {key_type} '{key}' provided{context}. "
        valid_options_message = f"Valid {key_type}s are {[k for k in default_config]}."
        raise ConfigurationError(error_message + valid_options_message)


def _validate_possible_age_differences(
    noise_type_config: Union[Dict, List],
    parameter: str,
    base_error_message: str,
) -> None:
    """
    Validates the user-provided values for the age-miswriting permutations
    parameter
    """
    if not isinstance(noise_type_config, (Dict, List)):
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must be a Dict or List. "
            f"Provided {noise_type_config} of type {type(noise_type_config)}."
        )
    for key in noise_type_config:
        if not isinstance(key, int):
            raise ConfigurationError(
                base_error_message + f"'{parameter}' must be a List of ints. "
                f"Provided {key} of type {type(key)}."
            )
        if key == 0:
            raise ConfigurationError(base_error_message + f"'{parameter}' cannot include 0.")
    if isinstance(noise_type_config, Dict):
        for value in noise_type_config.values():
            if not isinstance(value, (float, int)):
                raise ConfigurationError(
                    base_error_message
                    + f"'{parameter}' probabilities must be floats or ints. "
                    f"Provided {value} of type {type(value)}."
                )
            if not (0 <= value <= 1):
                raise ConfigurationError(
                    base_error_message
                    + f"'{parameter}' probabilities must be between 0 and 1 (inclusive). "
                    f"Provided {value} in {list(noise_type_config.values())}."
                )
        if not np.isclose(sum(noise_type_config.values()), 1.0):
            raise ConfigurationError(
                base_error_message + f"'{parameter}' probabilities must sum to 1. "
                f"Provided values sum to {sum(noise_type_config.values())}."
            )


def _validate_zipcode_digit_probabilities(
    noise_type_config: List, parameter: str, base_error_message: str
) -> None:
    """Validates the user-provided values for the zipcode digit noising probabilities"""
    if not isinstance(noise_type_config, List):
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must be a List. "
            f"Provided {noise_type_config} of type {type(noise_type_config)}."
        )
    if len(noise_type_config) != 5:
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must be a List of 5 probabilities. "
            f"{len(noise_type_config)} probabilities ({noise_type_config})."
        )
    for value in noise_type_config:
        _validate_probability(value, parameter, base_error_message)


def _validate_probability(
    noise_type_config: Union[int, float], parameter: str, base_error_message: str
) -> None:
    if not isinstance(noise_type_config, (float, int)):
        raise ConfigurationError(
            base_error_message + f"'{parameter}' probabilities must be floats or ints. "
            f"Provided {noise_type_config} of type {type(noise_type_config)}."
        )

    if not (0 <= noise_type_config <= 1):
        raise ConfigurationError(
            base_error_message + f"'{parameter}'s must be between 0 and 1 (inclusive). "
            f"Provided {noise_type_config}."
        )
