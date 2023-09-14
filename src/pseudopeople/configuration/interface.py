from pathlib import Path
from typing import Dict, Union

from pseudopeople.configuration.generator import get_configuration


def get_config(overrides: Union[Path, str, Dict] = None) -> Dict:
    """
    Function that returns the pseudopeople configuration including all
    default values. To get the default probability of nonresponse in the
    Decennial Census dataset:

    .. code-block:: pycon

        >>> import pseudopeople as psp
        >>> psp.get_config()['decennial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.0145}

    To view that same part of the configuration after applying a user
    override:

    .. code-block:: pycon

        >>> overrides = {'decennial_census': {'row_noise': {'do_not_respond': {'row_probability': 0.1}}}}
        >>> psp.get_config(overrides)['decenial_census']['row_noise']['do_not_respond']
        {'row_probability': 0.1}

    :param overrides:

        An optional override to the default configuration. Can be a
        Python dictionary, a path to a YAML file, or the special
        sentinel value `pseudopeople.NO_NOISE`, which will return a
        configuration where all configurable noise levels are set to 0.
        When passing a configuration dictionary or YAML configuration
        file, it is not necessary to provide a complete configuration;
        any configuration parameters not specified in `overrides` will
        be filled in with the default values.

    :return:

        Dictionary representing the configuration.

    :raises ConfigurationError:

        An invalid configuration is passed with overrides.

    """
    return get_configuration(overrides).to_dict()
