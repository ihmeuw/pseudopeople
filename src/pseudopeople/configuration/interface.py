from pathlib import Path
from typing import Dict, Union

from pseudopeople.configuration.generator import get_configuration


def get_config(overrides: Union[Path, str, Dict] = None) -> Dict:
    """
    Function that returns the pseudopeople configuration including all default values.
    To get the default probability of nonresponse in the Decennial Census dataset:

    .. code-block:: pycon

        >>> import pseudopeople as psp
        >>> psp.get_config()['decennial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.0145}

    To view that same part of the configuration after applying a user override:

    .. code-block:: pycon

        >>> overrides = {'decennial_census': {'row_noise': {'do_not_respond': {'row_probability': 0.1}}}}
        >>> psp.get_config(overrides)['decenial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.1}

    :param overrides: An optional override to the default configuration. Can be
        a path to a configuration YAML file or a dictionary. Passing a sentinel value
        of psp.NO_NOISE will override default values and return a configuration
        where all noise levels are set to 0.
    :return: Dictionary of the config.
    :raises ConfigurationError: An invalid configuration is passed with overrides.

    """
    return get_configuration(overrides).to_dict()
