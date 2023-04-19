"""
===========================
The Configuration Interface
===========================

An interface for users to interact with the configuration (config) in pseudopeople.

In ``pseudopeople`` a configuration is used to provide noise level that get passed to the noise functions to apply
various types of noise to given datasets.  Users can provide their own configuration that will override the default
values.
.. code-block:: python
>>> import pseudopeople as pp
>>> pp.get_config("decennial_census")
>>> user_config = {"decennial_census": {"row_noise": {"probability": {"omit_rows": 0.1},},},}
>>> pp.get_config("decennial_census", user_config)

"""

from pathlib import Path
from typing import Dict, Union

from loguru import logger

from pseudopeople.configuration.generator import get_configuration
from pseudopeople.schema_entities import DATASETS


def get_config(dataset_name: str = None, user_conf: Union[Path, str, Dict] = None) -> Dict:
    """
    Function that displays the configuration for the user
    :param dataset_name: Name of dataset to lookup in configuration.  Providing this argument returns the configuration
    for this specific form and no other forms in the configuration.  Possible dataset names include:
    ["american_communities_survey", "decennial_census", "current_population_survey", "social_security", "taxes_1040",
    "taxes_w2_and_1099", "women_infants_and_children"].
    :param user_conf: Dictionary of configuration values the user wishes to manually override.
    """

    config = get_configuration(user_conf)
    if dataset_name:
        if dataset_name in [dataset.name for dataset in DATASETS]:
            config = config[dataset_name]
        else:
            raise ValueError(
                f"'{dataset_name}' provided but is not a valid option for dataset type."
            )
    if user_conf and dataset_name not in user_conf.keys():
        logger.warning(
            f"'{dataset_name}' provided but is not in the user provided configuration."
        )

    return config.to_dict()
