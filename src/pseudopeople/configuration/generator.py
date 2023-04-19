from pathlib import Path
from typing import Dict, Union

import yaml
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.configuration.validator import validate_user_configuration
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASETS

# Define non-baseline default items
# NOTE: default values are defined in entity_types.RowNoiseType and entity_types.ColumnNoiseType
DEFAULT_NOISE_VALUES = {
    DATASETS.census.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omission.name: {
                Keys.PROBABILITY: 0.0145,
            }
        },
    },
    DATASETS.acs.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omission.name: {
                Keys.PROBABILITY: 0.0145,
            },
        },
    },
    DATASETS.cps.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omission.name: {
                Keys.PROBABILITY: 0.2905,
            },
        },
    },
    DATASETS.tax_w2_1099.name: {
        Keys.ROW_NOISE: {
            NOISE_TYPES.omission.name: {
                Keys.PROBABILITY: 0.005,
            },
        },
    },
    # No noise of any kind for SSN in the SSA observer
    DATASETS.ssa.name: {
        Keys.COLUMN_NOISE: {
            COLUMNS.ssn.name: {
                noise_type.name: {
                    (
                        Keys.PROBABILITY
                        if noise_type.probability is not None
                        else Keys.CELL_PROBABILITY
                    ): 0.0,
                }
                for noise_type in COLUMNS.ssn.noise_types
            }
        }
    },
}


def get_configuration(user_configuration: Union[Path, str, Dict] = None) -> ConfigTree:
    """
    Gets a noising configuration ConfigTree, optionally overridden by a user-provided YAML.

    :param user_configuration: A path to the YAML file or a dictionary defining user overrides for the defaults
    :return: a ConfigTree object of the noising configuration
    """

    noising_configuration = _generate_default_configuration()
    if user_configuration:
        add_user_configuration(noising_configuration, user_configuration)

    return noising_configuration


def _generate_default_configuration() -> ConfigTree:
    default_config_layers = [
        "baseline",
        "default",
        "user",
    ]
    noising_configuration = ConfigTree(layers=default_config_layers)
    # Instantiate the configuration file with baseline values
    baseline_dict = {}
    # Loop through each dataset
    for dataset in DATASETS:
        dataset_dict = {}
        row_noise_dict = {}
        column_dict = {}

        # Loop through row noise types
        for row_noise in dataset.row_noise_types:
            row_noise_type_dict = {}
            if row_noise.probability is not None:
                row_noise_type_dict[Keys.PROBABILITY] = row_noise.probability
            if row_noise_type_dict:
                row_noise_dict[row_noise.name] = row_noise_type_dict

        # Loop through columns and their applicable column noise types
        for column in dataset.columns:
            column_noise_dict = {}
            for noise_type in column.noise_types:
                column_noise_type_dict = {}
                if noise_type.probability is not None:
                    column_noise_type_dict[Keys.PROBABILITY] = noise_type.probability
                if noise_type.additional_parameters is not None:
                    for key, value in noise_type.additional_parameters.items():
                        column_noise_type_dict[key] = value
                if column_noise_type_dict:
                    # We should not have both 'probability' and 'cell_probability'
                    # TODO: move this into a pytest
                    if (
                        Keys.PROBABILITY in column_noise_type_dict
                        and Keys.CELL_PROBABILITY in column_noise_type_dict
                    ):
                        raise ValueError(
                            "'probability' and 'cell_probability' are mutually exclusive "
                            "but both are found in the default configuration for "
                            f"dataset '{dataset.name}', column '{column.name}', noise type '{noise_type.name}'"
                        )
                    column_noise_dict[noise_type.name] = column_noise_type_dict
            if column_noise_dict:
                column_dict[column.name] = column_noise_dict

        # Compile
        if row_noise_dict:
            dataset_dict[Keys.ROW_NOISE] = row_noise_dict
        if column_dict:
            dataset_dict[Keys.COLUMN_NOISE] = column_dict

        # Add the dataset's dictionary to baseline
        if dataset_dict:
            baseline_dict[dataset.name] = dataset_dict

    noising_configuration.update(baseline_dict, layer="baseline")

    # Update configuration with non-baseline default values
    noising_configuration.update(DEFAULT_NOISE_VALUES, layer="default")
    return noising_configuration


def add_user_configuration(
    noising_configuration: ConfigTree, user_configuration: Union[Path, str, Dict]
) -> None:
    if isinstance(user_configuration, (Path, str)):
        with open(user_configuration, "r") as f:
            user_configuration = yaml.full_load(f)

    validate_user_configuration(user_configuration, noising_configuration)

    user_configuration = _format_user_configuration(noising_configuration, user_configuration)
    noising_configuration.update(user_configuration, layer="user")


def _format_user_configuration(default_config: ConfigTree, user_dict: Dict) -> Dict:
    """Formats the user's configuration file as necessary, so it can properly
    update noising configuration to be used
    """
    user_dict = _format_age_miswriting_perturbations(default_config, user_dict)
    return user_dict


def _format_age_miswriting_perturbations(default_config: ConfigTree, user_dict: Dict) -> Dict:
    # Format any age perturbation lists as a dictionary with uniform probabilities
    for dataset in user_dict:
        user_perturbations = (
            user_dict[dataset]
            .get(Keys.COLUMN_NOISE, {})
            .get("age", {})
            .get(NOISE_TYPES.age_miswriting.name, {})
            .get(Keys.POSSIBLE_AGE_DIFFERENCES, {})
        )
        if not user_perturbations:
            continue
        formatted = {}
        default_perturbations = default_config[dataset][Keys.COLUMN_NOISE]["age"][
            NOISE_TYPES.age_miswriting.name
        ][Keys.POSSIBLE_AGE_DIFFERENCES]
        # Replace default configuration with 0 probabilities
        for perturbation in default_perturbations:
            formatted[perturbation] = 0
        if isinstance(user_perturbations, list):
            # Add user perturbations with uniform probabilities
            uniform_prob = 1 / len(user_perturbations)
            for perturbation in user_perturbations:
                formatted[perturbation] = uniform_prob
        else:
            for perturbation, prob in user_perturbations.items():
                formatted[perturbation] = prob

        user_dict[dataset][Keys.COLUMN_NOISE]["age"][NOISE_TYPES.age_miswriting.name][
            Keys.POSSIBLE_AGE_DIFFERENCES
        ] = formatted

    return user_dict
