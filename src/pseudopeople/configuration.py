from pathlib import Path
from typing import Dict, Union

import yaml
from vivarium.config_tree import ConfigTree

from pseudopeople.schema_entities import FORMS, NOISE_TYPES


class Keys:
    """Container for all non-form standard/repeated key names used in the configuration file"""

    ROW_NOISE = "row_noise"  # second layer, eg <form>: row_noise: {...}
    COLUMN_NOISE = "column_noise"  # second layer, eg <form>: column_noise: {...}
    PROBABILITY = "probability"
    ROW_NOISE_LEVEL = "row_noise_level"
    TOKEN_NOISE_LEVEL = "token_noise_level"


# Define non-baseline default items
# NOTE: default values are defined in entity_types.RowNoiseType and entity_types.ColumnNoiseType
DEFAULT_NOISE_VALUES = {
    FORMS.census.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omission.name: {
                Keys.PROBABILITY: 0.0145,
            }
        },
    },
    FORMS.acs.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omission.name: {
                Keys.PROBABILITY: 0.0145,
            },
        },
    },
    FORMS.cps.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omission.name: {
                Keys.PROBABILITY: 0.2905,
            },
        },
    },
}


def get_configuration(user_configuration: Union[Path, str, Dict] = None) -> ConfigTree:
    """
    Gets a noising configuration ConfigTree, optionally overridden by a user-provided YAML.

    :param user_configuration: A path to the YAML file or a dictionary defining user overrides for the defaults
    :return: a ConfigTree object of the noising configuration
    """

    default_config_layers = [
        "baseline",
        "default",
        "user",
    ]
    noising_configuration = ConfigTree(
        layers=default_config_layers,
    )

    # Instantiate the configuration file with baseline values
    baseline_dict = {}

    # Loop through each form
    for form in FORMS:
        form_dict = {}
        row_noise_dict = {}
        column_dict = {}

        # Loop through row noise types
        for row_noise in form.row_noise_types:
            row_noise_type_dict = {}
            if row_noise.probability is not None:
                row_noise_type_dict[Keys.PROBABILITY] = row_noise.probability
            if row_noise_type_dict:
                row_noise_dict[row_noise.name] = row_noise_type_dict

        # Loop through columns and their applicable column noise types
        for column in form.columns:
            column_noise_dict = {}
            for noise_type in column.noise_types:
                column_noise_type_dict = {}
                if noise_type.row_noise_level is not None:
                    column_noise_type_dict[Keys.ROW_NOISE_LEVEL] = noise_type.row_noise_level
                if noise_type.token_noise_level is not None:
                    column_noise_type_dict[
                        Keys.TOKEN_NOISE_LEVEL
                    ] = noise_type.token_noise_level
                if noise_type.additional_parameters is not None:
                    for key, value in noise_type.additional_parameters.items():
                        column_noise_type_dict[key] = value
                if column_noise_type_dict:
                    column_noise_dict[noise_type.name] = column_noise_type_dict
            if column_noise_dict:
                column_dict[column.name] = column_noise_dict

        # Compile
        if row_noise_dict:
            form_dict[Keys.ROW_NOISE] = row_noise_dict
        if column_dict:
            form_dict[Keys.COLUMN_NOISE] = column_dict

        # Add the form's dictionary to baseline
        if form_dict:
            baseline_dict[form.name] = form_dict

    noising_configuration.update(baseline_dict, layer="baseline")

    # Update configuration with non-baseline default values
    noising_configuration.update(DEFAULT_NOISE_VALUES, layer="default")

    # Update configuration with user-supplied values
    if user_configuration:
        if isinstance(user_configuration, (Path, str)):
            with open(user_configuration, "r") as f:
                user_configuration = yaml.full_load(f)
        user_configuration = format_user_configuration(
            user_configuration, noising_configuration
        )
        noising_configuration.update(user_configuration, layer="user")

    validate_noising_configuration(noising_configuration)

    return noising_configuration


def format_user_configuration(user_dict: Dict, default_config) -> Dict:
    """Formats the user's configuration file as necessary so it can properly
    update noising configuration to be used
    """
    user_dict = _format_age_miswriting_perturbations(user_dict, default_config)

    return user_dict


def _format_age_miswriting_perturbations(user_dict: Dict, default_config: ConfigTree) -> Dict:
    # Format any age perturbation lists as a dictionary with uniform probabilites
    for form in user_dict:
        user_perturbations = (
            user_dict[form]
            .get("column_noise", {})
            .get("age", {})
            .get("age_miswriting", {})
            .get("possible_perturbations", {})
        )
        if not user_perturbations:
            continue
        formatted = {}
        default_perturbations = default_config[form]["column_noise"]["age"]["age_miswriting"][
            "possible_perturbations"
        ]
        # Replace default configuration with 0 probabilities
        for perturbation in default_perturbations:
            formatted[perturbation] = 0
        if isinstance(user_perturbations, list):
            # Add user perturbations with uniform probabilities
            uniform_prob = 1 / len(user_perturbations)
            for perturbation in user_perturbations:
                formatted[perturbation] = uniform_prob
        elif isinstance(user_perturbations, dict):
            for perturbation, prob in user_perturbations.items():
                formatted[perturbation] = prob
        else:
            raise NotImplementedError(
                "age.age_miswriting.possible_perturbations can only be a list or dict, "
                f"received type {type(user_perturbations)}"
            )
        user_dict[form]["column_noise"]["age"]["age_miswriting"][
            "possible_perturbations"
        ] = formatted

    return user_dict


def validate_noising_configuration(config: ConfigTree) -> None:
    """Perform various validation checks on the final noising ConfigTree object"""
    _validate_age_miswriting(config)
    # TODO: validate omissions = [0, 0.5]


def _validate_age_miswriting(config: ConfigTree) -> None:
    possible_perturbations = _extract_values(config, "possible_perturbations")
    for form_perturbations in possible_perturbations:
        form_perturbations_dict = form_perturbations.to_dict()
        if 0 in form_perturbations_dict:
            # TODO: Find a way to report specific location in config file
            raise ValueError("Cannot include 0 in age_miswriting.possible_perturbations")
        if sum(form_perturbations_dict.values()) != 1:
            raise ValueError(
                "The provided possible_perturbation probabilities must sum to 1 but they "
                f"currently sum to {sum(form_perturbations_dict.values())}: {form_perturbations_dict}",
            )


def _extract_values(config: Union[ConfigTree, Dict], key: str):
    """Extract values with a specific key from a dict or configtree"""
    results = []
    for k, v in config.items():
        if k == key:
            results.append(v)
        if isinstance(v, (dict, ConfigTree)):
            for result in _extract_values(v, key):
                results.append(result)

    return results
