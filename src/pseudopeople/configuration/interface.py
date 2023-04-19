"""
===========================
The Configuration Interface
===========================

An interface for user's to interact with the configuration (config) in pseudopeople.

In ``pseudopeople`` a configuration is used to provide noise level that get passed to the noise functions to apply
various types of noise to given datasets.  Users can provide their own configuration that will override the deault
values.
.. code-block:: python
>>> import pseudopeople as pp
>>> pp.get_config("decennial_census")
>>> user_config = {"decennial_census": {"row_noise": {"probability": 0.1},},}
>>> pp.get_config("decennial_census", user_config)

"""

from pathlib import Path
from typing import Dict, Union

from loguru import logger

from pseudopeople.configuration.generator import get_configuration
from pseudopeople.constants.metadata import DATASET_NAMES


def get_config(
    dataset_name: str = None, user_configuration: Union[Path, str, Dict] = None
) -> Dict:
    """
    Function that displays the configuration for the user
    :param dataset_name: Name of dataset to lookup in configuration.  Providing this argument returns the configuration for
    this specific form and no other forms in the configuration.
    :param user_configuration: Dictionary of configuration values the user wishes to manually override.
    """

    config = get_configuration(user_configuration)
    if dataset_name:
        if dataset_name in DATASET_NAMES:
            config = config[dataset_name]
        elif dataset_name not in DATASET_NAMES:
            raise ValueError(
                f"{dataset_name} provided but is not a valid option for dataset type."
            )
    if user_configuration:
        if dataset_name not in user_configuration.keys():
            logger.warning(
                f"{dataset_name} provided but is not in the user provided configuration."
            )

    return config.to_dict()
