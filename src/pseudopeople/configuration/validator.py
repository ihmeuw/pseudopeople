from typing import Dict, List, Union

import numpy as np
from vivarium.config_tree import ConfigTree, ConfigurationKeyError

from pseudopeople.configuration import Keys


def validate_user_configuration(user_config: Dict, default_config: ConfigTree) -> None:
    """
    Validates the user-provided configuration. Confirms that all user-provided
    keys exist in the default configuration. Confirms that all user-provided
    values are valid for their respective noise functions.
    """
    for form, form_config in user_config.items():
        default_form_config = _get_default_config_node(default_config, form, "form")
        for key in form_config:
            _get_default_config_node(default_form_config, key, "configuration key", form)

        default_row_noise_config = default_form_config[Keys.ROW_NOISE]
        default_column_noise_config = default_form_config[Keys.COLUMN_NOISE]

        for noise_type, noise_type_config in form_config.get(Keys.ROW_NOISE, {}).items():
            default_noise_type_config = _get_default_config_node(
                default_row_noise_config, noise_type, "noise type", form
            )
            _validate_noise_type_config(
                noise_type_config, default_noise_type_config, form, noise_type
            )
            # TODO: validate omissions = [0, 0.5]

        for column, column_config in form_config.get(Keys.COLUMN_NOISE, {}).items():
            default_column_config = _get_default_config_node(
                default_column_noise_config, column, "column", form
            )
            for noise_type, noise_type_config in column_config.items():
                default_noise_type_config = _get_default_config_node(
                    default_column_config, noise_type, "noise type", form, column
                )
                _validate_noise_type_config(
                    noise_type_config, default_noise_type_config, form, noise_type, column
                )


def _validate_noise_type_config(
    noise_type_config: Union[Dict, List],
    default_noise_type_config: ConfigTree,
    form: str,
    noise_type: str,
    column: str = None,
) -> None:
    """
    Validates that all parameters are allowed for this noise function.
    Additionally, validates that the configuration values are permissible.
    """
    for parameter, parameter_config in noise_type_config.items():
        parameter_config_validator = {
            # todo add additional config value validators
            Keys.AGE_MISWRITING_PERTURBATIONS: _validate_age_miswriting_perturbations_config
        }.get(parameter, lambda *_: _)

        _ = _get_default_config_node(
            default_noise_type_config, parameter, "parameter", form, column, noise_type
        )
        parameter_config_validator(parameter_config, form, column)


def _get_default_config_node(
    default_config: ConfigTree,
    key: str,
    key_type: str,
    form: str = None,
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
        form_context = "" if form is None else f" for form '{form}'"
        column_context = "" if column is None else f" for column '{column}'"
        noise_type_context = "" if noise_type is None else f" and noise type '{noise_type}'"
        context = form_context + column_context + noise_type_context

        error_message = f"Invalid {key_type} '{key}' provided{context}. "
        valid_options_message = f"Valid {key_type}s are {[k for k in default_config]}."
        raise ValueError(error_message + valid_options_message)


def _validate_age_miswriting_perturbations_config(
    noise_type_config: Union[Dict, List], form: str, column: str
) -> None:
    """
    Validates the user-provided values for the age-miswriting permutations
    parameter
    """
    if not isinstance(noise_type_config, (Dict, List)):
        raise TypeError(
            "Invalid configuration type provided for age miswriting for form "
            f"{form} and column {column}."
        )

    for key in noise_type_config:
        if not isinstance(key, int):
            raise TypeError(
                "All possible age miswriting perturbations must be ints. "
                f"Provided {key} of type {type(key)} in the configuration "
                f"for form {form} and column {column}."
            )
        if key == 0:
            raise ValueError(
                "Cannot include 0 as an age miswriting perturbation. "
                f"Provided 0 in the configuration for form {form} and "
                f"column {column}."
            )

    if isinstance(noise_type_config, Dict):
        for value in noise_type_config.values():
            if not isinstance(value, float):
                raise TypeError(
                    "All possible age miswriting probabilities must be floats. "
                    f"Provided {value} of type {type(value)} in the configuration "
                    f"for form {form} and column {column}."
                )

        if not np.isclose(sum(noise_type_config.values()), 1.0):
            raise ValueError(
                "The provided age miswriting probabilities must sum to 1. "
                f"Provided values sum to {sum(noise_type_config.values())}."
            )
