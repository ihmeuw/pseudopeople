from pathlib import Path
import yaml
from vivarium.config_tree import ConfigTree
from typing import Dict, Union, NamedTuple
from pseudopeople.schema_entities import FORMS


class Keys(NamedTuple):
    """NamedTuple containing all key names used in the configuration file
    NOTE: 'additional_parameters' is actually a dict with its own key-values defined
    """
    ROW_NOISE = "row_noise"
    OMISSION = "omission"
    DUPLICATION = "duplication"
    COLUMN_NOISE = "column_noise"
    ROW_NOISE_LEVEL = "row_noise_level"
    TOKEN_NOISE_LEVEL = "token_noise_level"
    ADDITIONAL_PARAMETERS = "additional_parameters"

    
# Define non-baseline default items
# NOTE: default values are defined in entity_types.RowNoiseType and entity_types.ColumnNoiseType
DEFAULT_NOISE_VALUES = {
    # TODO: Uncomment when omission gets implemented
    # FORMS.CENSUS.name: {
    #     Keys.ROW_NOISE: {
    #         Keys.OMISSION: 0.0145,
    #     },
    # },
    # FORMS.ACS.name: {
    #     Keys.ROW_NOISE: {
    #         Keys.OMISSION: 0.0145,
    #     },
    # },
    # FORMS.CPS.name: {
    #     Keys.ROW_NOISE: {
    #         Keys.OMISSION: 0.2905,
    #     },
    # },
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
    for form in FORMS:
        if not form.is_implemented:
            continue
        baseline_dict[form.name] = {}
        baseline_dict[form.name][Keys.ROW_NOISE] = {}
        baseline_dict[form.name][Keys.COLUMN_NOISE] = {}
        for row_noise in form.row_noise_types:
            if not row_noise.is_implemented:
                continue
            if row_noise.noise_level is not None:
                baseline_dict[form.name][Keys.ROW_NOISE][row_noise.name] = row_noise.noise_level
        for column in form.columns:
            if not column.is_implemented:
                continue
            baseline_dict[form.name][Keys.COLUMN_NOISE][column.name] = {}
            for noise_type in column.noise_types:
                if not noise_type.is_implemented:
                    continue
                baseline_dict[form.name][Keys.COLUMN_NOISE][column.name][noise_type.name] = {}
                if noise_type.row_noise_level is not None:
                    baseline_dict[form.name][Keys.COLUMN_NOISE][column.name][noise_type.name][Keys.ROW_NOISE_LEVEL] = noise_type.row_noise_level
                if noise_type.token_noise_level is not None:
                    baseline_dict[form.name][Keys.COLUMN_NOISE][column.name][noise_type.name][Keys.TOKEN_NOISE_LEVEL] = noise_type.token_noise_level
                if noise_type.additional_parameters is not None:
                    for key, value in noise_type.additional_parameters.items():
                        baseline_dict[form.name][Keys.COLUMN_NOISE][column.name][noise_type.name][key] = value
        # Clean up empty layers that had no chance to `continue` out of a loop
        if not baseline_dict[form.name][Keys.ROW_NOISE]:
            del baseline_dict[form.name][Keys.ROW_NOISE]
        if not baseline_dict[form.name][Keys.COLUMN_NOISE]:
            del baseline_dict[form.name][Keys.COLUMN_NOISE]
    
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
        user_dict[form]["column_noise"]["age"]["age_miswriting"]["possible_perturbations"] = formatted

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
