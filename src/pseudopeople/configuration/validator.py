from typing import Callable, Dict, List, Union

import numpy as np
from loguru import logger
from vivarium.config_tree import ConfigTree, ConfigurationKeyError

from pseudopeople.configuration import Keys
from pseudopeople.constants import metadata
from pseudopeople.exceptions import ConfigurationError
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_scaling import get_options_for_column


def validate_overrides(overrides: Dict, default_config: ConfigTree) -> None:
    """
    Validates the user-provided overrides. Confirms that all user-provided
    keys exist in the default configuration. Confirms that all user-provided
    values are valid for their respective noise functions.
    """
    if not isinstance(overrides, Dict):
        raise ConfigurationError("Invalid configuration type provided.") from None
    for dataset, dataset_config in overrides.items():
        if not isinstance(dataset_config, Dict):
            raise ConfigurationError(
                f"'{dataset}' must be a Dict. "
                f"Provided {dataset_config} of type {type(dataset_config)}."
            )

        default_dataset_config = _get_default_config_node(default_config, dataset, "dataset")
        for key in dataset_config:
            _get_default_config_node(
                default_dataset_config, key, "configuration key", dataset
            )

        default_row_noise_config = default_dataset_config[Keys.ROW_NOISE]
        default_column_noise_config = default_dataset_config[Keys.COLUMN_NOISE]

        row_noise_config = dataset_config.get(Keys.ROW_NOISE, {})
        if not isinstance(row_noise_config, Dict):
            raise ConfigurationError(
                f"'{Keys.ROW_NOISE}' of '{dataset}' must be a Dict. "
                f"Provided {row_noise_config} of type {type(row_noise_config)}."
            )

        for noise_type, noise_type_config in row_noise_config.items():
            if not isinstance(noise_type_config, Dict):
                raise ConfigurationError(
                    f"Row noise type '{noise_type}' of dataset '{dataset}' must be a Dict. "
                    f"Provided {noise_type_config} of type {type(noise_type_config)}."
                )

            default_noise_type_config = _get_default_config_node(
                default_row_noise_config, noise_type, "noise type", dataset
            )
            _validate_noise_type_config(
                noise_type_config,
                default_noise_type_config,
                dataset,
                noise_type,
                DEFAULT_PARAMETER_CONFIG_VALIDATOR_MAP,
            )

        column_noise_config = dataset_config.get(Keys.COLUMN_NOISE, {})
        if not isinstance(column_noise_config, Dict):
            raise ConfigurationError(
                f"'{Keys.COLUMN_NOISE}' of '{dataset}' must be a Dict. "
                f"Provided {column_noise_config} of type {type(column_noise_config)}."
            )

        for column, column_config in column_noise_config.items():
            if not isinstance(column_config, Dict):
                raise ConfigurationError(
                    f"Column '{column}' of dataset '{dataset}' must be a Dict. "
                    f"Provided {column_config} of type {type(column_config)}."
                )

            default_column_config = _get_default_config_node(
                default_column_noise_config, column, "column", dataset
            )
            for noise_type, noise_type_config in column_config.items():
                if not isinstance(noise_type_config, Dict):
                    raise ConfigurationError(
                        f"Noise type '{noise_type}' of column '{column}' in dataset '{dataset}' must be a Dict. "
                        f"Provided {noise_type_config} of type {type(noise_type_config)}."
                    )

                default_noise_type_config = _get_default_config_node(
                    default_column_config, noise_type, "noise type", dataset, column
                )
                parameter_config_validator_map = {
                    NOISE_TYPES.use_nickname.name: {
                        Keys.CELL_PROBABILITY: _validate_nickname_probability
                    },
                    NOISE_TYPES.choose_wrong_option.name: {
                        Keys.CELL_PROBABILITY: lambda *args, **kwargs: _validate_choose_wrong_option_probability(
                            *args, **kwargs, column=column
                        )
                    },
                }.get(noise_type, DEFAULT_PARAMETER_CONFIG_VALIDATOR_MAP)
                _validate_noise_type_config(
                    noise_type_config,
                    default_noise_type_config,
                    dataset,
                    noise_type,
                    parameter_config_validator_map,
                    column,
                )


def _validate_noise_type_config(
    noise_type_config: Union[Dict, List],
    default_noise_type_config: ConfigTree,
    dataset: str,
    noise_type: str,
    parameter_config_validator_map: Dict[str, Callable],
    column: str = None,
) -> None:
    """
    Validates that all parameters are allowed for this noise function.
    Additionally, validates that the configuration values are permissible.
    """
    for parameter, parameter_config in noise_type_config.items():
        parameter_config_validator = parameter_config_validator_map.get(
            parameter, _validate_probability
        )

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
        raise ConfigurationError(error_message + valid_options_message) from None


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
    if len(noise_type_config) == 0:
        raise ConfigurationError(
            base_error_message + f"'{parameter}' must not be empty. "
            f"Provided {noise_type_config}."
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


def _validate_nickname_probability(
    noise_type_config: Union[int, float], parameter: str, base_error_message: str
) -> None:
    _validate_probability(noise_type_config, parameter, base_error_message)
    if noise_type_config > metadata.NICKNAMES_PROPORTION:
        logger.warning(
            base_error_message
            + f"The configured '{parameter}' is {noise_type_config}, but only approximately "
            f"{metadata.NICKNAMES_PROPORTION:.2%} of names have a nickname. "
            f"Replacing as many names with nicknames as possible."
        )


def _validate_choose_wrong_option_probability(
    noise_type_config: Union[int, float], parameter: str, base_error_message: str, column: str
):
    _validate_probability(noise_type_config, parameter, base_error_message)
    num_options = len(get_options_for_column(column))
    # The maximum: if the cell *selection* probability were set to 1, and every cell
    # selected an option uniformly at random, how many cells would actually change?
    # Each cell would have a 1 / num_options chance of staying the same.
    maximum_noise_probability = 1 - (1 / num_options)
    if noise_type_config > maximum_noise_probability:
        logger.warning(
            base_error_message
            + f"The configured '{parameter}' is {noise_type_config}, but pseudopeople "
            f"can only choose the wrong option with a maximum of {maximum_noise_probability:.5f} probability. "
            f"This maximum will be used instead of the configured value."
        )


DEFAULT_PARAMETER_CONFIG_VALIDATOR_MAP = {
    Keys.POSSIBLE_AGE_DIFFERENCES: _validate_possible_age_differences,
    Keys.ZIPCODE_DIGIT_PROBABILITIES: _validate_zipcode_digit_probabilities,
}
