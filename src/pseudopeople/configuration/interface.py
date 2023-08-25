from pathlib import Path
from typing import Dict, Union

import yaml
from loguru import logger

from pseudopeople.configuration.entities import NO_NOISE
from pseudopeople.configuration.generator import get_configuration
from pseudopeople.exceptions import ConfigurationError
from pseudopeople.schema_entities import DATASETS


def get_config(dataset_name: str = None, overrides: Union[Path, str, Dict] = None) -> Dict:
    """
    Function that returns the pseudopeople configuration,
    including all default values.
    If :code:`dataset_name` is None (the default), the returned dictionary includes
    the configuration for all datasets. If a dataset name is supplied, only that
    dataset's configuration is returned. In both cases, the returned dictionary has exactly
    the structure described on the :ref:`Configuration page <configuration_main>`.

    To get the default probability of nonresponse in the Decennial Census dataset:

    .. code-block:: pycon

        >>> import pseudopeople as psp
        >>> psp.get_config('decennial_census')['decennial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.0145}

    To view that same part of the configuration after applying a user override:

    .. code-block:: pycon

        >>> overrides = {'decennial_census': {'row_noise': {'do_not_respond': {'row_probability': 0.1}}}}
        >>> psp.get_config('decennial_census', overrides)['decennial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.1}

    :param dataset_name: An optional name of dataset to return the configuration
        for (defaults to all dataset configurations). Providing this argument returns
        the configuration for this specific dataset and no other datasets that exist
        in the configuration. Possible dataset names include:

            - "american_community_survey"
            - "decennial_census"
            - "current_population_survey"
            - "social_security"
            - "taxes_1040"
            - "taxes_w2_and_1099"
            - "women_infants_and_children"
    :param overrides: An optional override to the default configuration. Can be
        a path to a configuration YAML file or a dictionary. Passing a sentinel value
        of psp.NO_NOISE will override default values and return a configuration
        where all noise levels are set to 0.
    :return: Dictionary of the config.
    :raises ConfigurationError: An invalid configuration is passed with overrides.

    """
    if isinstance(overrides, (Path, str)) and overrides != NO_NOISE:
        with open(overrides, "r") as f:
            overrides = yaml.full_load(f)
    if isinstance(overrides, dict) and dataset_name not in overrides.keys():
        logger.warning(
            f"'{dataset_name}' provided but is not in the user provided configuration."
        )
    config = get_configuration(overrides).to_dict()
    if dataset_name:
        if dataset_name in [dataset.name for dataset in DATASETS]:
            config = {dataset_name: config[dataset_name]}
        else:
            raise ConfigurationError(
                f"'{dataset_name}' provided but is not a valid option for dataset type."
            )

    return config
