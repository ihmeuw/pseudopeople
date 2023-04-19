"""

"""
from pathlib import Path
from typing import Dict, Union

from loguru import logger

from pseudopeople.configuration.generator import get_configuration
from pseudopeople.constants.metadata import FORM_NAMES


def get_config(
    form_name: str = None, user_configuration: Union[Path, str, Dict] = None
) -> Dict:
    """
    Function that displays the configuration for the user
    :param form_name: Name of form to lookup in configuration.  Providing this argument returns the configuration for
      this specific form and no other forms in the configuration.
    :param user_configuration: Dictionary of configuration values the user wishes to manually override.
    """

    config = get_configuration(user_configuration)
    if form_name:
        if form_name in FORM_NAMES:
            config = config[form_name]
        elif form_name not in FORM_NAMES:
            raise ValueError(f"{form_name} provided but is not a valid option for form type.")
    if user_configuration:
        if form_name not in user_configuration.keys():
            logger.warning(
                f"{form_name} provided but is not in the user provided configuration."
            )

    return config.to_dict()
